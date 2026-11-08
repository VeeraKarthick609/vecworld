"""pytest plugin (loaded by conftest.py): a failing card_limit test releases its GPU memory (Task 10a re-review N3).

pytest keeps a failed test's traceback for the summary, and a traceback keeps its frames' locals: the spot walks'
TurboPainter and depth model, 3.3 GB of the 4 GB card. card_limit tests run first, so without this one flaky timing
failure runs every later GPU module out of memory and buries the real signal in ~100 errors. After the report is built
(its text is already rendered) the frames are cleared and the allocator's cache is emptied.
"""
import gc
import traceback

import pytest


def release_gpu_memory(excinfo) -> None:
    """Clear the locals of every frame in the exception's traceback (and its chained exceptions'), then give the
    freed device memory back to the driver."""
    seen, stack = set(), [excinfo.value]
    while stack:
        exc = stack.pop()
        if exc is None or id(exc) in seen:
            continue
        seen.add(id(exc))
        for frame, _ in traceback.walk_tb(exc.__traceback__):
            loc = frame.f_locals
            if type(loc) is dict:       # CPython <= 3.12: a snapshot dict pytest's repr made, outliving clear()
                loc.clear()
        traceback.clear_frames(exc.__traceback__)
        stack += [exc.__cause__, exc.__context__]
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if call.excinfo is not None and item.get_closest_marker("card_limit"):
        release_gpu_memory(call.excinfo)
    return outcome
