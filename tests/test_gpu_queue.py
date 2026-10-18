import threading

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
