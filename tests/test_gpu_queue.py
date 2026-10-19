import gc
import threading
import weakref

from vecworld.dream.gpu import GpuQueue


def _blocker(q, order):
    """Occupy the worker until the returned event is set."""
    started, release = threading.Event(), threading.Event()

    def busy():
        order.append("busy")
        started.set()
        assert release.wait(5)

    fut = q.submit(busy, "x")
    assert started.wait(5)
    return fut, release


def _rec(order, name):
    return lambda: order.append(name) or name


def test_on_demand_jumps_ahead_of_prefetch():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    pre = q.submit(_rec(order, "pre"), "a", prefetch=True)
    demand = q.submit(_rec(order, "demand"), "b")
    release.set()
    busy.result(timeout=5), demand.result(timeout=5), pre.result(timeout=5)
    assert order == ["busy", "demand", "pre"]


def test_new_prefetch_replaces_queued_one():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    first = q.submit(_rec(order, "p1"), "a", prefetch=True)
    second = q.submit(_rec(order, "p2"), "a", prefetch=True)
    assert first.cancelled() is True
    release.set()
    assert second.result(timeout=5) == "p2"
    assert order == ["busy", "p2"]


def test_prefetch_does_not_cancel_on_demand_or_other_owner():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    d = q.submit(_rec(order, "d"), "a")
    pb = q.submit(_rec(order, "pb"), "b", prefetch=True)
    pa = q.submit(_rec(order, "pa"), "a", prefetch=True)
    release.set()
    for f in (d, pb, pa):
        f.result(timeout=5)
    assert order == ["busy", "d", "pb", "pa"]


def test_owners_take_turns():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    futs = [q.submit(_rec(order, n), n[0]) for n in ("a1", "a2", "b1")]
    release.set()
    for f in futs:
        f.result(timeout=5)
    assert order == ["busy", "a1", "b1", "a2"]


def test_queue_survives_failing_job():
    q = GpuQueue()
    bad = q.submit(lambda: 1 / 0, "a")
    ok = q.submit(lambda: 7, "a")
    assert isinstance(bad.exception(timeout=2), ZeroDivisionError) and ok.result(timeout=2) == 7


def test_cancel_owner():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    a1 = q.submit(_rec(order, "a1"), "a")
    a2 = q.submit(_rec(order, "a2"), "a", prefetch=True)
    b1 = q.submit(_rec(order, "b1"), "b")
    q.cancel_owner("a")
    assert a1.cancelled() and a2.cancelled()
    release.set()
    assert b1.result(timeout=5) == "b1"
    assert order == ["busy", "b1"]


def test_externally_cancelled_future_never_runs():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    f = q.submit(_rec(order, "x"), "a")
    ok = q.submit(_rec(order, "ok"), "b")
    assert f.cancel()
    release.set()
    assert ok.result(timeout=5) == "ok"
    assert order == ["busy", "ok"]


def test_replaced_prefetch_keeps_turn_position():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    a1 = q.submit(_rec(order, "a1"), "a", prefetch=True)
    b1 = q.submit(_rec(order, "b1"), "b", prefetch=True)
    a2 = q.submit(_rec(order, "a2"), "a", prefetch=True)
    assert a1.cancelled()
    release.set()
    for f in (b1, a2):
        f.result(timeout=5)
    assert order == ["busy", "a2", "b1"]


def test_cancel_owner_of_running_job():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)  # owner "x" is running
    q.cancel_owner("x")
    later = q.submit(_rec(order, "later"), "x")
    release.set()
    assert busy.result(timeout=5) is None
    assert later.result(timeout=5) == "later"
    assert order == ["busy", "later"]


class _Payload:
    pass


def _submit_failing(q):
    payload = _Payload()
    ref = weakref.ref(payload)

    def boom():
        payload  # close over it
        raise RuntimeError("oom")

    fut = q.submit(boom, "a")
    assert isinstance(fut.exception(timeout=5), RuntimeError)
    return ref


def test_failed_job_is_released_while_worker_idle():
    q = GpuQueue()
    ref = _submit_failing(q)  # future dropped on return
    # worker is idle now; nothing else is submitted, so only the loop frame could hold it
    for _ in range(100):
        gc.collect()
        if ref() is None:
            break
        threading.Event().wait(0.01)
    assert ref() is None


def _set(q, owner, names, order):
    return q.submit_set([(n, _rec(order, n)) for n in names], owner)


def test_set_runs_heading_first():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    futs = _set(q, "a", ["heading", "left", "right"], order)
    assert list(futs) == ["heading", "left", "right"]
    release.set()
    for f in futs.values():
        f.result(timeout=5)
    assert order == ["busy", "heading", "left", "right"]


def test_resubmit_cancels_only_changed_queued_key():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    first = _set(q, "a", ["h", "l", "r"], order)
    second = _set(q, "a", ["h", "r", "r2"], order)
    assert second["h"] is first["h"] and second["r"] is first["r"]
    assert first["l"].cancelled()
    assert not first["h"].cancelled() and not first["r"].cancelled()
    release.set()
    for f in second.values():
        f.result(timeout=5)
    assert order == ["busy", "h", "r", "r2"]


def test_resubmit_reorders_to_new_set_order():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    _set(q, "a", ["h", "l", "r"], order)
    second = _set(q, "a", ["r", "h", "l"], order)
    release.set()
    for f in second.values():
        f.result(timeout=5)
    assert order == ["busy", "r", "h", "l"]


def test_running_job_is_never_cancelled_and_is_reused():
    q, order = GpuQueue(), []
    started, release = threading.Event(), threading.Event()

    def slow():
        order.append("h")
        started.set()
        assert release.wait(5)
        return "h"

    first = q.submit_set([("h", slow), ("l", _rec(order, "l"))], "a")
    assert started.wait(5)
    # heading is running: a set without it must not cancel it; one with it reuses it
    q.submit_set([("x", _rec(order, "x"))], "a")
    assert not first["h"].cancelled() and first["h"].running()
    again = q.submit_set([("h", _rec(order, "h2")), ("y", _rec(order, "y"))], "a")
    assert again["h"] is first["h"]
    assert first["l"].cancelled()
    release.set()
    assert first["h"].result(timeout=5) == "h"
    again["y"].result(timeout=5)
    assert order == ["h", "y"]


def test_finished_key_can_be_requeued():
    q, order = GpuQueue(), []
    f1 = _set(q, "a", ["h"], order)["h"]
    assert f1.result(timeout=5) == "h"
    f2 = _set(q, "a", ["h"], order)["h"]
    assert f2 is not f1 and f2.result(timeout=5) == "h"
    assert order == ["h", "h"]


def test_externally_cancelled_key_is_requeued():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    f = _set(q, "a", ["h"], order)["h"]
    assert f.cancel()
    g = _set(q, "a", ["h"], order)["h"]
    assert g is not f
    release.set()
    assert g.result(timeout=5) == "h"
    assert order == ["busy", "h"]


def test_empty_set_cancels_queued_and_leaves_queue_usable():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    futs = _set(q, "a", ["h", "l"], order)
    assert q.submit_set([], "a") == {}
    assert all(f.cancelled() for f in futs.values())
    later = q.submit(_rec(order, "later"), "b", prefetch=True)
    release.set()
    assert later.result(timeout=5) == "later"
    assert order == ["busy", "later"]


def test_set_does_not_touch_other_owner_or_demand():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    d = q.submit(_rec(order, "d"), "a")
    pb = _set(q, "b", ["b1"], order)
    pa = _set(q, "a", ["a1"], order)
    _set(q, "a", ["a2"], order)
    assert pa["a1"].cancelled() and not pb["b1"].cancelled() and not d.cancelled()
    release.set()
    d.result(timeout=5), pb["b1"].result(timeout=5)
    assert order[:3] == ["busy", "d", "b1"]


def test_demand_jumps_ahead_of_set_and_owners_take_turns():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    sa = _set(q, "a", ["a1", "a2"], order)
    sb = _set(q, "b", ["b1", "b2"], order)
    d = q.submit(_rec(order, "d"), "c")
    release.set()
    for f in (*sa.values(), *sb.values(), d):
        f.result(timeout=5)
    assert order == ["busy", "d", "a1", "b1", "a2", "b2"]


def test_single_prefetch_does_not_keep_previous_with_same_name():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    f1 = q.submit(_rec(order, "p1"), "a", prefetch=True)
    f2 = q.submit(_rec(order, "p2"), "a", prefetch=True)
    assert f1.cancelled() and not f2.cancelled()
    release.set()
    assert f2.result(timeout=5) == "p2"


def test_done_future_in_finish_window_is_not_reused(monkeypatch):
    q, order = GpuQueue(), []
    entered, go = threading.Event(), threading.Event()
    orig = q._finish

    def slow_finish(fut, rk):
        entered.set()
        assert go.wait(5)
        orig(fut, rk)

    monkeypatch.setattr(q, "_finish", slow_finish)
    bad = q.submit_set([("h", lambda: 1 / 0)], "a")["h"]
    assert entered.wait(5)
    assert bad.done()  # resolved, but the worker has not dropped its key yet
    retry = q.submit_set([("h", _rec(order, "h"))], "a")["h"]
    assert retry is not bad
    go.set()
    assert retry.result(timeout=5) == "h"
    assert isinstance(bad.exception(timeout=5), ZeroDivisionError)


def test_set_of_three_vs_one_is_fair():
    q, order = GpuQueue(), []
    busy, release = _blocker(q, order)
    sa = _set(q, "a", ["a1", "a2", "a3"], order)
    sb = _set(q, "b", ["b1"], order)
    release.set()
    for f in (*sa.values(), *sb.values()):
        f.result(timeout=5)
    assert order == ["busy", "a1", "b1", "a2", "a3"]


def _release_check(job_factory):
    q = GpuQueue()
    payload = _Payload()
    ref = weakref.ref(payload)
    fut = q.submit_set([("k", job_factory(payload))], "a")["k"]
    fut.exception(timeout=5)
    del payload, fut, job_factory
    for _ in range(100):
        gc.collect()
        if ref() is None:
            break
        threading.Event().wait(0.01)
    return ref


def test_set_job_released_after_success():
    assert _release_check(lambda p: (lambda: p and 1))() is None


def test_set_job_released_after_failure():
    def factory(p):
        def boom():
            p
            raise RuntimeError("oom")
        return boom
    assert _release_check(factory)() is None
