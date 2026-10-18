"""Single-worker GPU queue: on-demand before prefetch, round-robin per owner."""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from concurrent.futures import Future
from typing import Callable, TypeVar

T = TypeVar("T")

_DEMAND, _PREFETCH = 0, 1


class GpuQueue:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        # priority -> owner -> deque[(future, fn)]; OrderedDict order = turn order
        self._queues: tuple[OrderedDict, OrderedDict] = (OrderedDict(), OrderedDict())
        self._thread = threading.Thread(target=self._run, name="gpu-queue", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], T], owner: str, prefetch: bool = False) -> "Future[T]":
        fut: Future = Future()
        with self._cond:
            q = self._queues[_PREFETCH if prefetch else _DEMAND]
            if prefetch:
                for old, _ in q.pop(owner, ()):
                    old.cancel()
            q.setdefault(owner, deque()).append((fut, fn))
            self._cond.notify()
        return fut

    def cancel_owner(self, owner: str) -> None:
        with self._cond:
            for q in self._queues:
                for fut, _ in q.pop(owner, ()):
                    fut.cancel()

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

    def _run(self) -> None:
        while True:
            with self._cond:
                item = self._next()
                while item is None:
                    self._cond.wait()
                    item = self._next()
            fut, fn = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001 - job errors go to the future
                fut.set_exception(e)
