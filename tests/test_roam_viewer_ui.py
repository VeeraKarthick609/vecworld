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


LIMIT = 0.55 * 4     # world.js: the walkable radius


def open_page(d, url, auto=1, **query):
    extra = "".join(f"&{k}={v}" for k, v in query.items())
    d.get(f"{url}?auto={auto}&clear={VOID}{extra}")
    wait(d, "window.roam && window.roam.ready")


def walk_chain(d, last, timeout=120):
    """Follow the auto-walk until spot `last` is current and its fade is done; returns the spot indices seen."""
    seen = [js(d, "roam.spotIndex")]
    end = time.time() + timeout
    while time.time() < end and not js(d, f"roam.spotIndex === {last} && !roam.fading"):
        i = js(d, "roam.spotIndex")
        if i != seen[-1]:
            seen.append(i)
        time.sleep(0.02)
    if js(d, "roam.spotIndex") != seen[-1]:
        seen.append(js(d, "roam.spotIndex"))
    return seen


def hold(d, key, seconds):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).key_down(key).pause(seconds).key_up(key).perform()


def test_page_loads_and_builds_every_spot_in_the_worker(site, browser):
    open_page(browser, site, auto=0)
    assert js(browser, "roam.built") == 2            # spot 0 and its placed child, each meshed by the worker
    assert js(browser, "roam.workerBuilt") == 2
    assert js(browser, "roam.triangles") > 10000      # a real depth mesh, not an empty one
    assert js(browser, "roam.spotIndex") == 0
    assert js(browser, "window.roamErrors") == []


def test_roam_state_is_read_only(site, browser):
    open_page(browser, site, auto=0)
    browser.execute_script("try { roam.spotIndex = 5 } catch (e) {}")
    assert js(browser, "roam.spotIndex") == 0


def test_auto_walk_crosses_both_transitions(site, browser):
    open_page(browser, site)
    assert walk_chain(browser, 2) == [0, 1, 2]
    pos = js(browser, "roam.pos")
    assert set(pos) == {"x", "z"}
    # the chain heads north (0 deg) then 60 deg east of north: well north and east of where it started
    wait(browser, "roam.walked > 2.0")
    assert js(browser, "roam.world.z") < -1.0 and js(browser, "roam.world.x") > 0.3
    assert js(browser, "window.roamErrors") == []


def test_no_void_anywhere_while_looking_around(site, browser):
    open_page(browser, site, auto=0)
    for i in range(3):
        browser.execute_async_script("roamDev.goto(arguments[0]).then(arguments[1])", i)
        assert js(browser, "roam.spotIndex") == i and not js(browser, "roam.fading")
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


def test_strafing_follows_the_camera(site, browser):
    """Camera looking north (yaw 0): D walks east (+x), A west, W north (−z)."""
    open_page(browser, site, auto=0)
    browser.execute_script("roamDev.look(0, 0.18)")
    for key, axis, sign in (("d", "x", 1), ("a", "x", -1), ("w", "z", -1)):
        before = js(browser, "roam.pos")
        hold(browser, key, 0.6)
        wait(browser, "roamDev.hero.speed < 0.01")
        after = js(browser, "roam.pos")
        moved = after[axis] - before[axis]
        assert sign * moved > 0.1, f"{key}: {axis} moved {moved:+.3f}"
        other = "z" if axis == "x" else "x"
        assert abs(after[other] - before[other]) < 0.5 * abs(moved), f"{key}: drifted along {other}"
    assert js(browser, "window.roamErrors") == []


def test_far_spots_switch_at_the_edge(site, browser):
    """Spots further apart than twice the walkable radius: the bisector is out of reach, so the switch happens at
    the edge, and the walk goes on (no being pinned there)."""
    open_page(browser, site, distance=6)
    assert walk_chain(browser, 2) == [0, 1, 2]
    sw = js(browser, "roam.switches")
    assert [(s["from"], s["to"]) for s in sw] == [(0, 1), (1, 2)]
    assert all(s["at"] >= 0.98 * LIMIT - 0.01 for s in sw), sw
    assert js(browser, "window.roamErrors") == []


def test_edge_without_a_spot_ahead_is_reported(site, browser):
    """At the last spot nothing lies ahead: walking out holds at the edge and roam.edge says so (fog goes there)."""
    open_page(browser, site, auto=0)
    browser.execute_async_script("roamDev.goto(2).then(arguments[0])")
    from selenium.webdriver.common.action_chains import ActionChains
    ActionChains(browser).key_down("w").perform()
    try:
        wait(browser, "roam.edge !== null", timeout=30)
        r = js(browser, "Math.hypot(roam.pos.x, roam.pos.z)")
        assert 0.98 * LIMIT - 0.01 <= r <= LIMIT + 1e-6
        assert js(browser, "roam.spotIndex") == 2
    finally:
        ActionChains(browser).key_up("w").perform()
    wait(browser, "roam.edge === null")


def test_spots_are_disposed_along_a_long_walk(site, browser):
    open_page(browser, site, chain=6)
    wait(browser, "roam.spotIndex === 1 && !roam.fading", timeout=90)
    wait(browser, "roam.built === 3")          # spot 1, the spot faded from, and spot 2 placed ahead
    early = js(browser, "roam.memory")
    assert walk_chain(browser, 5, timeout=240)[-1] == 5
    assert js(browser, "roam.maxBuilt") <= 3
    late = js(browser, "roam.memory")
    assert late["geometries"] <= early["geometries"] and late["textures"] <= early["textures"], (early, late)
    assert js(browser, "roam.workerBuilt") == 6
    assert js(browser, "window.roamErrors") == []


def test_rescaled_child_switches_without_a_jump(site, browser):
    """Children at scale 0.88: the scenery doesn't move on screen at the switch, the view scale takes the ratio
    and then eases back to 1, and the walk still reaches the next spot."""
    open_page(browser, site, scale=0.88)
    assert walk_chain(browser, 2) == [0, 1, 2]
    sw = js(browser, "roam.switches")
    assert len(sw) == 2
    for s in sw:
        assert abs(s["dx"]) < 1.0 and abs(s["dy"]) < 1.0, s     # pixels
        assert s["k"] < 0.95                                    # the re-root took the ratio into the view scale
    wait(browser, "Math.abs(roam.k - 1) < 0.02", timeout=60)


def test_worker_reply_for_an_unknown_job_is_an_error(site, browser):
    open_page(browser, site, auto=0)
    browser.execute_script("roamDev.world.worker.onmessage({ data: { id: 12345 } })")
    errs = js(browser, "window.roamErrors")
    assert len(errs) == 1 and "unknown job 12345" in errs[0]
