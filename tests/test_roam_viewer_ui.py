"""Browser tests for the Dream Roam viewer core: the transition dev page over the stand-in fixture spots
(headless Firefox, a static server over web/)."""
import functools
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vecworld.__main__ import WEB_DIR

pytestmark = pytest.mark.ui

VOID = "ff00ff"     # the dev page's clear colour in tests: any pixel of it on screen is a hole in the world


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def site():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB_DIR)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/roam/transition.html"
    srv.shutdown()


@pytest.fixture
def browser():
    from selenium import webdriver
    from selenium.webdriver.firefox.service import Service

    opts = webdriver.FirefoxOptions()
    opts.add_argument("-headless")
    for k in ("webgl.force-enabled", "gfx.x11-egl.force-enabled"):
        opts.set_preference(k, True)
    opts.set_preference("webgl.out-of-process", False)
    # headless Firefox has no GLX display: WebGL through surfaceless EGL on Mesa's software renderer (llvmpipe)
    env = {**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1", "EGL_PLATFORM": "surfaceless", "MOZ_WEBGL_FORCE_EGL": "1"}
    d = webdriver.Firefox(options=opts, service=Service("/snap/bin/geckodriver", env=env))
    d.set_window_size(800, 600)
    yield d
    d.quit()


def js(d, expr):
    return d.execute_script(f"return {expr}")


def wait(d, expr, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if js(d, expr):
            return
        assert not js(d, "window.roamErrors || []"), js(d, "window.roamErrors")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for: {expr}; roam = {js(d, 'JSON.stringify(window.roam)')}")


def open_page(d, url, auto=1):
    d.get(f"{url}?auto={auto}&clear={VOID}")
    wait(d, "window.roam && window.roam.ready")


def test_page_loads_and_builds_every_spot_in_the_worker(site, browser):
    open_page(browser, site, auto=0)
    assert js(browser, "roam.built") == 3            # the three fixture spots, each meshed by the worker
    assert js(browser, "roam.workerBuilt") == 3
    assert js(browser, "roam.triangles") > 10000      # a real depth mesh, not an empty one
    assert js(browser, "roam.spotIndex") == 0
    assert js(browser, "window.roamErrors") == []


def test_roam_state_is_read_only(site, browser):
    open_page(browser, site, auto=0)
    browser.execute_script("try { roam.spotIndex = 5 } catch (e) {}")
    assert js(browser, "roam.spotIndex") == 0


def test_auto_walk_crosses_both_transitions(site, browser):
    open_page(browser, site)
    seen = [js(browser, "roam.spotIndex")]
    end = time.time() + 90
    while time.time() < end and seen[-1] != 2:
        i = js(browser, "roam.spotIndex")
        if i != seen[-1]:
            seen.append(i)
        time.sleep(0.02)
    assert seen == [0, 1, 2]
    pos = js(browser, "roam.pos")
    assert set(pos) == {"x", "z"}
    # the chain heads north (0 deg) then 60 deg east of north: well north and east of where it started
    wait(browser, "roam.walked > 2.0")
    assert js(browser, "roam.world.z") < -1.0 and js(browser, "roam.world.x") > 0.3
    assert js(browser, "window.roamErrors") == []


def test_no_void_anywhere_while_looking_around(site, browser):
    open_page(browser, site, auto=0)
    for i in range(3):
        browser.execute_script(f"roamDev.goto({i})")
        wait(browser, f"roam.spotIndex === {i} && !roam.fading")
        for yaw in range(0, 360, 45):
            for pitch in (-0.1, 0.45):
                n = browser.execute_script(f"return roamDev.voidPixels({yaw}, {pitch})")
                assert n == 0, f"spot {i} yaw {yaw} pitch {pitch}: {n} void pixels"


def test_no_void_during_the_transitions(site, browser):
    open_page(browser, site)
    worst = 0
    end = time.time() + 90
    while time.time() < end and not js(browser, "roam.spotIndex === 2 && !roam.fading"):
        worst = max(worst, browser.execute_script("return roamDev.voidPixels()"))
    assert js(browser, "roam.spotIndex") == 2
    assert worst == 0
