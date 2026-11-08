// The CUDA side of the weight slots' sequencing (engine.slots, engine.stream; docs/engine/10-streamer.md).
//
// Every PyTorch op called from Python releases the GIL around its work and takes it back afterwards. With another
// Python thread busy, taking it back waits for that thread's switch interval (5 ms): a tensor copy_, an event record,
// a stream wait each cost milliseconds instead of microseconds, and BlockSlots made ~25 of them per block (a copy per
// tensor). SlotRing makes the same driver calls from one extension call that keeps the GIL: a per-block hook is a few
// microseconds whatever the other threads do.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("SlotRing: ") + what + ": " + cudaGetErrorString(e));
    }
}

// Per slot s: copied[s], recorded on the side stream after the copy into s; freed[s], recorded on the compute stream
// after the block that read s. load waits freed (once the slot was used) before copying, before() makes the compute
// stream wait copied.
//
// Every load also gets a ticket (1, 2, 3, ... in issue order) and its own event, so a reader can ask whether THAT copy
// has finished reading its source (done(ticket)), however the slots have been reused since. The side stream runs its
// copies in order, so tickets complete in order: done() retires finished events from the front of the queue and
// keeps the highest finished ticket. The ticket is assigned inside load(), under the GIL, together with the record.
class SlotRing {
  public:
    SlotRing(int64_t ring, int64_t side_stream, int64_t device)
        : device_(static_cast<c10::DeviceIndex>(device)), side_(reinterpret_cast<cudaStream_t>(side_stream)) {
        TORCH_CHECK(ring >= 1, "SlotRing needs ring >= 1, got ", ring);
        c10::cuda::CUDAGuard g(device_);
        for (int64_t s = 0; s < ring; ++s) {
            cudaEvent_t a, b;
            check(cudaEventCreateWithFlags(&a, cudaEventDisableTiming), "cudaEventCreate");
            check(cudaEventCreateWithFlags(&b, cudaEventDisableTiming), "cudaEventCreate");
            copied_.push_back(a);
            freed_.push_back(b);
        }
    }
    ~SlotRing() {
        for (auto e : copied_) cudaEventDestroy(e);
        for (auto e : freed_) cudaEventDestroy(e);
        for (auto& p : pending_) cudaEventDestroy(p.second);
        for (auto e : spare_) cudaEventDestroy(e);
    }
    SlotRing(const SlotRing&) = delete;
    SlotRing& operator=(const SlotRing&) = delete;

    // copies: flat (device dst, host src, bytes) triples, all into slot s, after its last user if wait_freed; returns
    // the load's ticket
    int64_t load(int64_t s, bool wait_freed, const std::vector<int64_t>& copies) {
        TORCH_CHECK(copies.size() % 3 == 0, "SlotRing.load: copies must be (dst, src, bytes) triples");
        c10::cuda::CUDAGuard g(device_);
        if (wait_freed) check(cudaStreamWaitEvent(side_, freed_.at(s), 0), "cudaStreamWaitEvent");
        for (size_t k = 0; k < copies.size(); k += 3) {
            check(cudaMemcpyAsync(reinterpret_cast<void*>(copies[k]), reinterpret_cast<const void*>(copies[k + 1]),
                                  static_cast<size_t>(copies[k + 2]), cudaMemcpyHostToDevice, side_),
                  "cudaMemcpyAsync");
        }
        check(cudaEventRecord(copied_.at(s), side_), "cudaEventRecord");
        cudaEvent_t t;
        if (spare_.empty()) {
            check(cudaEventCreateWithFlags(&t, cudaEventDisableTiming), "cudaEventCreate");
        } else {
            t = spare_.back();
            spare_.pop_back();
        }
        check(cudaEventRecord(t, side_), "cudaEventRecord");
        pending_.emplace_back(++issued_, t);
        return issued_;
    }
    // whether load ticket's copy has finished (it no longer reads its source); a ticket not issued yet is not done
    bool done(int64_t ticket) {
        while (!pending_.empty()) {
            cudaError_t e = cudaEventQuery(pending_.front().second);
            if (e == cudaErrorNotReady) {
                cudaGetLastError();
                break;
            }
            check(e, "cudaEventQuery");
            finished_ = pending_.front().first;
            spare_.push_back(pending_.front().second);
            pending_.pop_front();
        }
        return ticket <= finished_;
    }
    int64_t issued() const { return issued_; }
    // the current (compute) stream waits until slot s holds what was last loaded into it
    void before(int64_t s) {
        c10::cuda::CUDAGuard g(device_);
        check(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device_).stream(), copied_.at(s), 0),
              "cudaStreamWaitEvent");
    }
    // slot s is free once the work queued so far on the current stream ends
    void after(int64_t s) {
        c10::cuda::CUDAGuard g(device_);
        check(cudaEventRecord(freed_.at(s), c10::cuda::getCurrentCUDAStream(device_).stream()), "cudaEventRecord");
    }
    // whether slot s's last copy has finished (it no longer reads its source)
    bool copied(int64_t s) {
        cudaError_t e = cudaEventQuery(copied_.at(s));
        if (e == cudaErrorNotReady) {
            cudaGetLastError();
            return false;
        }
        check(e, "cudaEventQuery");
        return true;
    }

  private:
    c10::DeviceIndex device_;
    cudaStream_t side_;
    std::vector<cudaEvent_t> copied_, freed_;
    std::deque<std::pair<int64_t, cudaEvent_t>> pending_;  // (ticket, its event) of copies not known finished
    std::vector<cudaEvent_t> spare_;
    int64_t issued_ = 0, finished_ = 0;
};

bool pinned_at(int64_t ptr) {
    cudaPointerAttributes a;
    cudaError_t e = cudaPointerGetAttributes(&a, reinterpret_cast<const void*>(ptr));
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return a.type == cudaMemoryTypeHost;
}

// whether host bytes [ptr, ptr + nbytes) are page-locked (cudaHostAlloc'd or cudaHostRegister'ed): a copy from them is
// asynchronous. Checks the first and the last byte (a buffer starting inside a registered range may run past it).
bool host_pinned(int64_t ptr, int64_t nbytes) {
    return pinned_at(ptr) && (nbytes <= 1 || pinned_at(ptr + nbytes - 1));
}

// Launch an instantiated CUDA graph (torch.cuda.CUDAGraph.raw_cuda_graph_exec()) on the current stream, keeping the
// GIL: CUDAGraph.replay() releases it (and, under contention, waits a switch interval to get it back). replay() also
// runs the graph's RNG prologue, which the FLUX blocks' graphs don't need (they draw no random numbers).
void graph_launch(int64_t exec) {
    check(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(exec), c10::cuda::getCurrentCUDAStream().stream()),
          "cudaGraphLaunch");
}

}  // namespace

void register_stream(pybind11::module& m) {
    namespace py = pybind11;
    py::class_<SlotRing>(m, "SlotRing",
                         "CUDA events and copies for `ring` weight slots on a side stream; every method keeps the GIL")
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("ring"), py::arg("side_stream"), py::arg("device"))
        .def("load", &SlotRing::load, py::arg("slot"), py::arg("wait_freed"), py::arg("copies"))
        .def("before", &SlotRing::before, py::arg("slot"))
        .def("after", &SlotRing::after, py::arg("slot"))
        .def("copied", &SlotRing::copied, py::arg("slot"))
        .def("done", &SlotRing::done, py::arg("ticket"))
        .def("issued", &SlotRing::issued);
    m.def("host_pinned", &host_pinned, "whether host bytes [ptr, ptr + nbytes) are page-locked", py::arg("ptr"),
          py::arg("nbytes") = 1);
    m.def("graph_launch", &graph_launch, "cudaGraphLaunch(exec, current stream), keeping the GIL", py::arg("exec"));
}
