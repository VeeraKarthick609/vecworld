"""Single-worker GPU queue: on-demand before prefetch, round-robin per owner."""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from concurrent.futures import Future
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")

_DEMAND, _PREFETCH = 0, 1


class GpuQueue:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        # priority -> owner -> deque[(future, fn, running_key)]; OrderedDict order = turn order
        self._queues: tuple[OrderedDict, OrderedDict] = (OrderedDict(), OrderedDict())
        # (owner, key) -> future, for keyed prefetch jobs queued or running
        self._keyed: dict[tuple[str, Hashable], Future] = {}
        self._thread = threading.Thread(target=self._run, name="gpu-queue", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], T], owner: str, prefetch: bool = False) -> "Future[T]":
        if prefetch:
            # a one-job set under a fresh key, so it always replaces the owner's queued jobs
            return next(iter(self.submit_set([(object(), fn)], owner).values()))
        fut: Future = Future()
        with self._cond:
            self._queues[_DEMAND].setdefault(owner, deque()).append((fut, fn, None))
            self._cond.notify()
        return fut

    def submit_set(
        self, jobs: "list[tuple[Hashable, Callable[[], T]]]", owner: str
    ) -> "dict[Hashable, Future[T]]":
        """Make `jobs` (in priority order) the owner's prefetch set.

        Queued jobs whose key is not in the set are cancelled; a queued or running job
        whose key is in the set is kept and its future returned. Running jobs are never
        cancelled. The owner keeps its place in the round-robin rotation.

        Key contract: a repeated key reuses the original queued/running job AND its fn,
        ignoring the new fn. Keys must therefore encode everything the fn depends on
        (e.g. world version + heading). Finished (done or failed) jobs are never reused.
        """
        result: dict[Hashable, Future] = {}
        with self._cond:
            wanted = {}
            for key, fn in jobs:
                wanted.setdefault(key, fn)
            q = self._queues[_PREFETCH]
            old = q.get(owner)
            queued = {rk[1]: (item) for item in (old or ()) if (rk := item[2]) is not None}
            items = []
            for key, fn in wanted.items():
                kept = self._keyed.get((owner, key))
                if kept is not None and not kept.done():  # done covers cancelled; never reuse a finished job
                    result[key] = kept
                    if key in queued:  # still waiting: keep it, in the new order
                        items.append(queued.pop(key))
                    continue
                fut: Future = Future()
                rk = (owner, key)
                self._keyed[rk] = fut
                result[key] = fut
                items.append((fut, fn, rk))
            for fut, _, rk in queued.values():  # queued but no longer wanted
                fut.cancel()
                if self._keyed.get(rk) is fut:
                    del self._keyed[rk]
            if items:
                if old is None:
                    old = q[owner] = deque()
                old.clear()
                old.extend(items)
                self._cond.notify()
            elif old is not None:
                del q[owner]
        return result

    def cancel_owner(self, owner: str) -> None:
        with self._cond:
            for q in self._queues:
                for fut, _, rk in q.pop(owner, ()):
                    fut.cancel()
                    if rk is not None and self._keyed.get(rk) is fut:
                        del self._keyed[rk]

    def _next(self):
        # caller holds the lock
        for q in self._queues:
            if q:
                owner, dq = next(iter(q.items()))
                item = dq.popleft()
                if dq:
                    q.move_to_end(owner)
                else:
                    del q[owner]
                return item
        return None

    def _finish(self, fut, rk) -> None:
        if rk is not None:
            with self._cond:
                if self._keyed.get(rk) is fut:
                    del self._keyed[rk]

    def _run(self) -> None:
        while True:
            with self._cond:
                item = self._next()
                while item is None:
                    self._cond.wait()
                    item = self._next()
            fut, fn, rk = item
            try:
                if fut.set_running_or_notify_cancel():
                    try:
                        fut.set_result(fn())
                    except BaseException as e:  # noqa: BLE001 - job errors go to the future
                        fut.set_exception(e)
            finally:
                self._finish(fut, rk)
            # drop references so a finished/failed job's tensors can be freed
            del item, fut, fn, rk
