// Reads from a file straight into memory with O_DIRECT through io_uring, on the raw syscalls (Task 10b,
// docs/engine/11-io-uring.md). liburing's headers are not installed here; the kernel's <linux/io_uring.h> is all it
// takes: io_uring_setup makes the two rings, mmap maps them into this process, io_uring_enter submits and waits.
//
//   submission ring (SQ): we write a read request (an SQE: fd, file offset, length, destination address) and move the
//                         tail; io_uring_enter(to_submit) hands them to the kernel, which starts the reads and returns
//   completion ring (CQ): the kernel writes one CQE per finished read (its result: bytes read, or -errno) and moves
//                         the tail; we read them and move the head. No syscall is needed to look.
//
// O_DIRECT reads skip the page cache: the device DMAs into the destination itself, so the offset, the length and the
// destination address must be multiples of the logical block size (ALIGN = 4096 here, a multiple of every NVMe's).
// A request is split into chunks of `chunk` bytes; at most queue_depth chunks are in the kernel at once, the rest wait
// here and go in as earlier ones complete (on done() / wait()).
//
// One thread at a time (the GPU worker): the rings are not locked. wait() and read_into() release the GIL while they
// block; submit() and done() keep it (they never wait).
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t ALIGN = 4096;

// Raised as Python's OSError (IOError).
struct IOFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string errstr(int e) { return std::string(strerror(e)); }

int uring_setup(unsigned entries, io_uring_params* p) {
    return static_cast<int>(syscall(__NR_io_uring_setup, entries, p));
}
int uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return static_cast<int>(syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0));
}

class DirectReader {
  public:
    // direct=false opens the file without O_DIRECT (through the page cache): a test switch, to show that the
    // page-cache checks can fail.
    DirectReader(const std::string& path, int64_t queue_depth, int64_t chunk, bool direct) : path_(path) {
        TORCH_CHECK(queue_depth >= 1 && queue_depth <= 4096, "DirectReader: queue_depth must be 1..4096, got ",
                    queue_depth);
        TORCH_CHECK(chunk >= ALIGN && chunk % ALIGN == 0 && chunk <= (1LL << 30),
                    "DirectReader: chunk must be a multiple of 4096 up to 1 GiB, got ", chunk);
        qd_ = static_cast<unsigned>(queue_depth);
        chunk_ = chunk;
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
        if (fd_ < 0) throw IOFailure("DirectReader: open(" + path + ", O_DIRECT): " + errstr(errno));
        struct stat st;
        if (fstat(fd_, &st) == 0) size_ = st.st_size;
        io_uring_params p;
        memset(&p, 0, sizeof p);
        ring_ = uring_setup(qd_, &p);
        if (ring_ < 0) {
            int e = errno;
            ::close(fd_);
            throw IOFailure("DirectReader: io_uring_setup: " + errstr(e));
        }
        sq_entries_ = p.sq_entries;
        size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        single_ = p.features & IORING_FEAT_SINGLE_MMAP;
        if (single_) sq_sz = cq_sz = std::max(sq_sz, cq_sz);
        sq_sz_ = sq_sz;
        cq_sz_ = cq_sz;
        sq_ptr_ = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_SQ_RING);
        if (sq_ptr_ == MAP_FAILED) fail_setup("mmap SQ ring");
        cq_ptr_ = single_ ? sq_ptr_
                          : mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_,
                                 IORING_OFF_CQ_RING);
        if (cq_ptr_ == MAP_FAILED) fail_setup("mmap CQ ring");
        sqes_sz_ = p.sq_entries * sizeof(io_uring_sqe);
        sqes_ = static_cast<io_uring_sqe*>(mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                                                ring_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) fail_setup("mmap SQEs");
        char* sq = static_cast<char*>(sq_ptr_);
        sq_tail_ = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
        sq_mask_ = *reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
        sq_array_ = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
        char* cq = static_cast<char*>(cq_ptr_);
        cq_head_ = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
        cq_tail_ = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
        cq_mask_ = *reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
        cqes_ = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
        if (qd_ > sq_entries_) qd_ = sq_entries_;
        slots_.resize(qd_);
        for (unsigned k = 0; k < qd_; ++k) free_.push_back(qd_ - 1 - k);
    }
    ~DirectReader() {
        try {
            close();
        } catch (...) {
        }
    }
    DirectReader(const DirectReader&) = delete;
    DirectReader& operator=(const DirectReader&) = delete;

    int64_t size() const { return size_; }
    int64_t queue_depth() const { return qd_; }
    int64_t chunk() const { return chunk_; }
    int64_t inflight() const { return inflight_; }
    int64_t queued() const { return static_cast<int64_t>(queue_.size()); }
    int64_t completions() const { return completions_; }

    // Test hook: the nth chunk completion from now (1 = the next) is reported with result `res` instead of its own
    // (a negative errno, or a short byte count). The read itself still ran; only what we're told about it changes.
    void inject_fault(int64_t nth, int64_t res) {
        fault_at_ = completions_ + nth;
        fault_res_ = static_cast<int>(res);
    }
    // Test hook: the next io_uring_enter while reaping completions fails with errno err (a hard error such as EFAULT).
    void inject_enter_fault(int err) { enter_fault_ = err; }

    // Start reading [offset, offset + nbytes) into dst (all three multiples of 4096). The request succeeds once its
    // first `need` bytes are in (a read may run past the end of the file, as the aligned tail of its last block does).
    // Returns the request's id for done() / wait().
    int64_t submit(int64_t offset, int64_t nbytes, int64_t need, int64_t dst) {
        check_open();
        if (offset < 0 || nbytes <= 0 || offset % ALIGN || nbytes % ALIGN || dst % ALIGN) {
            throw std::invalid_argument("DirectReader.submit: offset " + std::to_string(offset) + ", nbytes " +
                                        std::to_string(nbytes) + " and dst must be multiples of 4096 (nbytes > 0)");
        }
        if (need < 0 || need > nbytes) throw std::invalid_argument("DirectReader.submit: need must be 0..nbytes");
        int64_t id = ++issued_;
        Req& r = reqs_[id];
        r.offset = offset;
        r.end = offset + need;
        for (int64_t o = 0; o < nbytes; o += chunk_) {
            int64_t n = std::min(chunk_, nbytes - o);
            queue_.push_back(Chunk{id, offset + o, dst + o, n});
            ++r.left;
        }
        push();
        return id;
    }

    // Whether request id has finished (reaps completions and tops the kernel's queue up; never blocks). A failed
    // request raises OSError naming its offset, every time it is asked about; a finished one is forgotten once reported.
    bool done(int64_t id) {
        check_open();
        reap(false);
        push();
        return report(id);
    }

    // Block until request id has finished (the GIL released meanwhile); raises as done() does.
    void wait(int64_t id) {
        check_open();
        {
            pybind11::gil_scoped_release nogil;
            drive(id);
        }
        report(id);
    }

    // Read file bytes [offset, offset + nbytes) into dst[0, nbytes), blocking (GIL released). Any offset and length:
    // a start or end that isn't 4096-aligned is read as the aligned superset into a bounce buffer, then copied out.
    // Raises OSError naming the offset if the file ends before offset + nbytes.
    void read_into(int64_t offset, int64_t nbytes, torch::Tensor dst) {
        check_open();
        TORCH_CHECK(dst.device().is_cpu() && dst.is_contiguous(), "DirectReader.read_into: dst must be contiguous CPU");
        TORCH_CHECK(offset >= 0 && nbytes >= 0, "DirectReader.read_into: offset and nbytes must be >= 0");
        TORCH_CHECK(dst.numel() * dst.element_size() >= nbytes, "DirectReader.read_into: dst holds ",
                    dst.numel() * dst.element_size(), " bytes, ", nbytes, " asked");
        if (nbytes == 0) return;
        char* out = static_cast<char*>(dst.data_ptr());
        int64_t lo = offset / ALIGN * ALIGN, hi = (offset + nbytes + ALIGN - 1) / ALIGN * ALIGN;
        int64_t cap = dst.numel() * dst.element_size();
        // straight into dst only when the aligned superset is exactly the range (else dst[nbytes:] would be written)
        bool direct = lo == offset && offset + nbytes == hi && reinterpret_cast<uintptr_t>(out) % ALIGN == 0 &&
                      cap >= hi - lo;
        char* buf = out;
        if (!direct) {
            buf = static_cast<char*>(std::aligned_alloc(ALIGN, hi - lo));
            if (!buf) throw std::bad_alloc();
        }
        try {
            int64_t id = submit(lo, hi - lo, offset + nbytes - lo, reinterpret_cast<int64_t>(buf));
            wait(id);
            if (!direct) memcpy(out, buf + (offset - lo), nbytes);
        } catch (...) {
            if (!direct) std::free(buf);
            throw;
        }
        if (!direct) std::free(buf);
    }

    // Wait for every read in flight (their destinations must outlive them), then unmap the rings and close the file.
    // Safe twice.
    void close() {
        if (ring_ < 0) return;
        queue_.clear();
        {
            pybind11::gil_scoped_release nogil;
            while (inflight_ > 0) {
                if (reap(true) < 0) break;
            }
        }
        munmap(sqes_, sqes_sz_);
        if (!single_) munmap(cq_ptr_, cq_sz_);
        munmap(sq_ptr_, sq_sz_);
        ::close(ring_);
        ::close(fd_);
        ring_ = fd_ = -1;
        reqs_.clear();
    }

  private:
    struct Chunk {
        int64_t req, off, dst, len;
    };
    struct Req {
        int64_t offset = 0, end = 0, left = 0;
        bool failed = false;
        std::string err;
    };

    [[noreturn]] void fail_setup(const char* what) {
        int e = errno;
        if (sqes_ && sqes_ != MAP_FAILED) munmap(sqes_, sqes_sz_);
        if (!single_ && cq_ptr_ && cq_ptr_ != MAP_FAILED) munmap(cq_ptr_, cq_sz_);
        if (sq_ptr_ && sq_ptr_ != MAP_FAILED) munmap(sq_ptr_, sq_sz_);
        ::close(ring_);
        ::close(fd_);
        ring_ = fd_ = -1;
        throw IOFailure(std::string("DirectReader: ") + what + ": " + errstr(e));
    }
    void check_open() const {
        if (ring_ < 0) throw IOFailure("DirectReader on " + path_ + " is closed");
        if (!broken_.empty()) throw IOFailure("DirectReader on " + path_ + " is broken: " + broken_);
    }

    // Move queued chunks into the submission ring while the kernel has room, and submit them.
    void push() {
        unsigned n = 0;
        unsigned tail = *sq_tail_;
        while (!queue_.empty() && !free_.empty()) {
            Chunk c = queue_.front();
            queue_.pop_front();
            unsigned slot = free_.back();
            free_.pop_back();
            slots_[slot] = c;
            unsigned idx = tail & sq_mask_;
            io_uring_sqe* s = &sqes_[idx];
            memset(s, 0, sizeof *s);
            s->opcode = IORING_OP_READ;
            s->fd = fd_;
            s->addr = static_cast<uint64_t>(c.dst);
            s->len = static_cast<uint32_t>(c.len);
            s->off = static_cast<uint64_t>(c.off);
            s->user_data = slot;
            sq_array_[idx] = idx;
            ++tail;
            ++n;
        }
        if (!n) return;
        __atomic_store_n(sq_tail_, tail, __ATOMIC_RELEASE);
        unsigned left = n;
        while (left) {
            int r = uring_enter(ring_, left, 0, 0);
            if (r < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EBUSY) {    // the kernel wants completions reaped first
                    reap(false);
                    continue;
                }
                // SQEs are published but not taken: the ring no longer matches our books; refuse every later call
                broken_ = "io_uring_enter (submit): " + errstr(errno);
                throw IOFailure("DirectReader: " + broken_);
            }
            left -= static_cast<unsigned>(r);
            inflight_ += r;
        }
    }

    // Read the completion ring. block: wait for at least one completion first (if any read is in flight).
    // Returns how many completed, or -1 if nothing is in flight.
    int reap(bool block) {
        if (inflight_ == 0) return -1;
        unsigned head = *cq_head_;
        unsigned tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        if (head == tail || enter_fault_) {
            // nothing posted yet: let the kernel run pending completion work (and wait for one if block)
            int r, e;
            if (enter_fault_) {                     // test hook (_inject_enter_fault): this call fails with it
                r = -1;
                e = enter_fault_;
                enter_fault_ = 0;
            } else {
                r = uring_enter(ring_, 0, block ? 1 : 0, IORING_ENTER_GETEVENTS);
                e = errno;
            }
            if (r < 0 && e != EINTR && e != EAGAIN && e != EBUSY && e != ETIME) {
                // chunks may still be in the kernel, their destinations no longer tracked by any request the caller
                // waits on: refuse every later call (as a hard submit error does), so none is read into again (N1)
                broken_ = "io_uring_enter (wait): " + errstr(e);
                throw IOFailure("DirectReader: " + broken_);
            }
            tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        }
        int got = 0;
        while (head != tail) {
            io_uring_cqe* c = &cqes_[head & cq_mask_];
            int res = c->res;
            if (++completions_ == fault_at_) res = fault_res_;
            complete(static_cast<unsigned>(c->user_data), res);
            ++head;
            ++got;
        }
        __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
        return got;
    }

    void complete(unsigned slot, int res) {
        Chunk c = slots_[slot];
        free_.push_back(slot);
        --inflight_;
        auto it = reqs_.find(c.req);
        if (it == reqs_.end()) return;          // closed / forgotten
        Req& r = it->second;
        if (res < 0) {
            fail(c.req, r, "read of " + std::to_string(c.len) + " bytes at offset " + std::to_string(c.off) + ": " +
                        errstr(-res));
        } else if (res < c.len && c.off + res < r.end) {
            if (res == 0 || res % 512) {            // end of file (or a torn tail) before the bytes asked for
                fail(c.req, r, "short read at offset " + std::to_string(c.off + res) + ": the file ends before byte " +
                            std::to_string(r.end) + " (request at offset " + std::to_string(r.offset) + ")");
            } else {                                // a partial read: the rest goes again, still aligned
                queue_.push_front(Chunk{c.req, c.off + res, c.dst + res, c.len - res});
                return;
            }
        }
        --r.left;
    }

    // The request fails as a whole: its chunks still queued here are dropped (never submitted, so nothing more of it
    // lands in its buffer); those in the kernel are waited for (left counts them) before anyone is told.
    void fail(int64_t id, Req& r, const std::string& why) {
        if (!r.failed) {
            r.failed = true;
            r.err = "DirectReader " + path_ + ": " + why;
        }
        auto keep = std::remove_if(queue_.begin(), queue_.end(), [&](const Chunk& q) { return q.req == id; });
        r.left -= static_cast<int64_t>(queue_.end() - keep);
        queue_.erase(keep, queue_.end());
    }

    // GIL released: reap and top up until every chunk of request id has completed or been dropped (a failed request
    // too: nothing of it is in flight when this returns).
    void drive(int64_t id) {
        for (;;) {
            auto it = reqs_.find(id);
            if (it == reqs_.end() || it->second.left == 0) return;
            push();
            if (reap(true) < 0 && queue_.empty()) {
                throw std::logic_error("DirectReader: request " + std::to_string(id) + " has chunks left but none "
                                       "in flight or queued");
            }
        }
    }

    bool report(int64_t id) {
        auto it = reqs_.find(id);
        if (it == reqs_.end()) {
            if (id >= 1 && id <= issued_) return true;   // finished and reported before
            throw std::invalid_argument("DirectReader: no request " + std::to_string(id));
        }
        Req& r = it->second;
        if (r.left > 0) return false;
        if (r.failed) throw IOFailure(r.err);
        reqs_.erase(it);
        return true;
    }

    std::string path_;
    int fd_ = -1, ring_ = -1;
    int64_t size_ = -1, chunk_ = 0, issued_ = 0, inflight_ = 0, completions_ = 0, fault_at_ = 0;
    int fault_res_ = 0, enter_fault_ = 0;
    std::string broken_;
    unsigned qd_ = 0, sq_entries_ = 0;
    bool single_ = false;
    void *sq_ptr_ = nullptr, *cq_ptr_ = nullptr;
    size_t sq_sz_ = 0, cq_sz_ = 0, sqes_sz_ = 0;
    io_uring_sqe* sqes_ = nullptr;
    unsigned *sq_tail_ = nullptr, *sq_array_ = nullptr, *cq_head_ = nullptr, *cq_tail_ = nullptr;
    unsigned sq_mask_ = 0, cq_mask_ = 0;
    io_uring_cqe* cqes_ = nullptr;
    std::vector<Chunk> slots_;          // the chunk each in-flight SQE reads (user_data indexes it)
    std::vector<unsigned> free_;        // free indices into slots_
    std::deque<Chunk> queue_;           // chunks waiting for room in the kernel's queue
    std::unordered_map<int64_t, Req> reqs_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    py::register_exception<IOFailure>(m, "DirectReadError", PyExc_OSError);
    py::class_<DirectReader>(m, "DirectReader",
                             "O_DIRECT reads of one file through io_uring (raw syscalls); one thread at a time")
        .def(py::init<const std::string&, int64_t, int64_t, bool>(), py::arg("path"), py::arg("queue_depth") = 8,
             py::arg("chunk") = 1 << 20, py::arg("direct") = true)
        .def("submit", &DirectReader::submit, py::arg("offset"), py::arg("nbytes"), py::arg("need"), py::arg("dst"))
        .def("done", &DirectReader::done, py::arg("id"))
        .def("wait", &DirectReader::wait, py::arg("id"))
        .def("read_into", &DirectReader::read_into, py::arg("offset"), py::arg("nbytes"), py::arg("dst"))
        .def("close", &DirectReader::close)
        .def_property_readonly("size", &DirectReader::size)
        .def_property_readonly("queue_depth", &DirectReader::queue_depth)
        .def_property_readonly("chunk", &DirectReader::chunk)
        .def_property_readonly("inflight", &DirectReader::inflight)
        .def_property_readonly("queued", &DirectReader::queued)
        .def_property_readonly("completions", &DirectReader::completions)
        .def("_inject_fault", &DirectReader::inject_fault, py::arg("nth"), py::arg("res"))
        .def("_inject_enter_fault", &DirectReader::inject_enter_fault, py::arg("err"));
}
