"""Browser tests for the Dream Roam game viewer, against the real app running fake models (headless Firefox,
software WebGL)."""
import math
import socket
import threading
import time

import pytest
import uvicorn
from conftest import FakeDepth, FakeDreamer, FakePainter, firefox

from vecworld.__main__ import WEB_DIR
from vecworld.api.app import create_app
from vecworld.config import MOODS
from vecworld.roam.service import RoamService

pytestmark = pytest.mark.ui

# 8b: the walkable radius comes from each spot (the server's walk limits), read from the page as roam.rim; with
# the fake depth it is ~4.5 everywhere (was a fixed LIMIT = 2.2).


class UiPainter(FakePainter):
    """Solid panoramas. While .gate is clear, every step reports .report through its progress hook (as FluxPainter
    does, when .report is set), then blocks (and sets .entered); .fail_steps makes every step a GPU out-of-memory;
    .pano_delay slows the first panorama."""

    def __init__(self):
        super().__init__()
        self.fail_steps = False
        self.pano_delay = 0.0
        self.progress = None            # the service's hook
        self.report = None              # (done, total) a blocked step reports
        self.gate, self.entered = threading.Event(), threading.Event()
        self.gate.set()

    def panorama(self, *a, **kw):
        if self.pano_delay and self.report is not None:
            self.progress(*self.report)
        time.sleep(self.pano_delay)
        return super().panorama(*a, **kw)

    def step(self, *a, **kw):
        if not self.gate.is_set():
            if self.report is not None:
                self.progress(*self.report)
            self.entered.set()
            assert self.gate.wait(60)
        if self.fail_steps:
            self.fail_next = True
        return super().step(*a, **kw)


class RecordingService(RoamService):
    """Records every call the viewer makes as (name, argument, monotonic time). With .hide_next_image, the next
    /next's panorama file is moved aside before the viewer can load it (.hidden = (aside, original))."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[tuple[str, object, float]] = []
        self.walked: list[float | None] = []
        self.hide_next_image = False
        self.hidden = None
        self.refuse_enter = False   # every /enter answers unknown_spot, as if the spot's files were gone

    def _rec(self, name, arg):
        self.calls.append((name, arg, time.monotonic()))

    def start(self, sid, place, moods):
        self._rec("start", (place, list(moods)))
        return super().start(sid, place, moods)

    def next(self, sid, exit_heading, walked=None):
        self._rec("next", exit_heading)
        self.walked.append(walked)
        v = super().next(sid, exit_heading, walked)
        if self.hide_next_image:
            self.hide_next_image = False
            aside = v.image_path.with_suffix(".aside")
            v.image_path.rename(aside)
            self.hidden = (aside, v.image_path)
        return v

    def enter(self, sid, spot_id):
        self._rec("enter", spot_id)
        if self.refuse_enter:
            from vecworld.roam.errors import UnknownSpot

            raise UnknownSpot("That spot has faded from the dream.")
        return super().enter(sid, spot_id)

    def heading(self, sid, heading):
        self._rec("heading", heading)
        return super().heading(sid, heading)

    def set_theme(self, sid, place):
        self._rec("theme", place)
        return super().set_theme(sid, place)

    def set_moods(self, sid, moods):
        self._rec("moods", list(moods))
        return super().set_moods(sid, moods)

    def since(self, n, name=None):
        return [(a, t) if name else (k, a) for k, a, t in self.calls[n:] if name in (None, k)]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import torch

    mp = pytest.MonkeyPatch()
    mp.setattr(torch.cuda, "empty_cache", lambda: None)   # a fake out-of-memory must not touch the real GPU
    painter = UiPainter()
    # No places refill: the neighbours stay the fallback ones, so every test sees the same names.
    svc = RecordingService(painter, FakeDepth(), FakeDreamer(), tmp_path_factory.mktemp("scenes"),
                           spawn=lambda fn: None)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(svc, WEB_DIR), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", svc, painter
    srv.should_exit = True
    painter.gate.set()
    mp.undo()


@pytest.fixture(autouse=True)
def healthy(server):
    _, _, painter = server
    painter.fail_steps, painter.fail_next, painter.pano_delay, painter.report = False, False, 0.0, None
    server[1].hide_next_image = False
    server[1].refuse_enter = False
    painter.gate.set()
    painter.entered.clear()
    yield
    painter.gate.set()


@pytest.fixture
def browser():
    d = firefox(1000, 700)
    yield d
    d.quit()


def js(d, expr):
    return d.execute_script(f"return {expr}")


def wait(d, expr, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if js(d, expr):
            return
        assert not js(d, "window.roamErrors || []"), js(d, "window.roamErrors")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for: {expr}; roam = {js(d, 'JSON.stringify(window.roam)')}")


def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


TOAST = "!document.getElementById('toast').hidden"


def submit(d, text):
    from selenium.webdriver.common.keys import Keys

    box = d.find_element("id", "place")
    box.clear()
    box.send_keys(text, Keys.ENTER)


def open_game(d, url):
    d.get(url)
    wait(d, "window.roam && roam.ready")


def start(d, url, place="a quiet forest"):
    open_game(d, url)
    submit(d, place)
    wait(d, "roam.started && roam.spotId !== null")


def hold(d, key, seconds):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).key_down(key).pause(seconds).key_up(key).perform()


def key_down(d, key):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).key_down(key).perform()


def key_up(d, key):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).key_up(key).perform()


def radius(d):
    return js(d, "Math.hypot(roam.pos.x, roam.pos.z)")


def rim(d):
    """The current spot's walkable radius at the player's bearing."""
    return js(d, "roam.rim")


def ang(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def test_start_shows_the_world_and_chips_equal_moods(server, browser):
    url, svc, _ = server
    open_game(browser, url)
    assert js(browser, "[...document.querySelectorAll('.chip')].map(c => c.dataset.mood)") == list(MOODS)
    assert js(browser, "roam.started") is False
    assert not js(browser, "document.getElementById('intro').hidden")
    browser.find_element("css selector", ".chip[data-mood='night']").click()
    n = len(svc.calls)
    submit(browser, "a quiet forest")
    wait(browser, "roam.started && roam.spotId !== null")
    assert svc.since(n) == [("start", ("a quiet forest", ["night"]))]
    assert js(browser, "roam.place") == "a quiet forest"
    assert js(browser, "document.getElementById('intro').hidden")
    assert js(browser, "roam.built") == 1
    # The fake panorama is solid navy: the world on screen must be showing it.
    r, g, b = js(browser, "roamDev.pixel(0.5, 0.15)")
    assert b > 90 and r < 40 and g < 40, (r, g, b)
    assert js(browser, "window.roamErrors") == []


def test_wasd_moves_the_character(server, browser):
    url, svc, _ = server
    start(browser, url)
    browser.execute_script("roamDev.look(0, 0.18)")
    for key, axis, sign in (("w", "z", -1), ("d", "x", 1), ("s", "z", 1), ("a", "x", -1)):
        before = js(browser, "roam.pos")
        hold(browser, key, 0.5)
        wait(browser, "roamDev.hero.speed < 0.01")
        moved = js(browser, f"roam.pos.{axis}") - before[axis]
        assert sign * moved > 0.1, f"{key}: {axis} moved {moved:+.3f}"


def test_keys_are_ignored_while_typing(server, browser):
    url, _, _ = server
    start(browser, url)
    box = browser.find_element("id", "place")
    box.click()
    box.send_keys("wasd")
    assert box.get_attribute("value") == "wasd"
    assert js(browser, "roam.pos") == {"x": 0, "z": 0}


def test_dragging_turns_the_camera(server, browser):
    from selenium.webdriver.common.action_chains import ActionChains

    url, _, _ = server
    start(browser, url)
    yaw = js(browser, "roam.camYaw")
    canvas = browser.find_element("css selector", "#stage canvas")
    ActionChains(browser).move_to_element(canvas).click_and_hold().move_by_offset(-150, 0) \
        .move_by_offset(-150, 0).release().perform()
    turned = js(browser, "roam.camYaw") - yaw
    assert turned > 0.5, turned          # dragging left turns the view right (yaw grows)
    assert js(browser, "window.roamErrors") == []


def stalls(samples, walk_speed):
    """Spans (ms) during which the walk clip played (weight > 0.3) while the character barely moved."""
    out, since = [], None
    for x in samples:
        bad = x["walk"] > 0.3 and x["ground"] < 0.05 * walk_speed
        if bad and since is None:
            since = x["t"]
        elif not bad and since is not None:
            out.append(x["t"] - since)
            since = None
    if since is not None:
        out.append(samples[-1]["t"] - since)
    return out


def test_edge_waits_in_fog_then_continues(server, browser):
    url, svc, painter = server
    painter.gate.clear()                 # every step blocks: the pre-dreams around the first spot never finish
    start(browser, url)
    first = js(browser, "roam.spotId")
    browser.execute_script("roamDev.look(0, 0.18)")
    n = len(svc.calls)
    key_down(browser, "w")
    try:
        wait(browser, "roam.stepping")
        wait_for(lambda: [h for h, _ in svc.since(n, "next")], timeout=5)
        assert ang(svc.since(n, "next")[0][0], 0) < 5
        assert radius(browser) < rim(browser) - 0.3   # asked for ahead of the rim (by about the coast + LEAD)
        assert painter.entered.wait(10)
        # Coasts to a stop just inside the rim, in the fog, with the feet slowing as the ground speed does.
        wait(browser, "roam.fog && roam.braking && roamDev.hero.speed < 0.02", timeout=10)
        time.sleep(0.5)
        edge = rim(browser)
        assert 0.9 * edge <= radius(browser) <= edge + 1e-6
        # the walked distance sent is where the player leaves the spot: its rim that way
        assert svc.walked[-1] == pytest.approx(edge, abs=0.05)
        assert js(browser, "roam.spotId") == first
        wait(browser, "Number(getComputedStyle(document.getElementById('fog')).opacity) > 0.5")
        walk_speed = js(browser, "roamDev.hero.walkSpeed")
        assert max(stalls(js(browser, "roam.samples"), walk_speed), default=0) <= 100
        painter.gate.set()
        wait(browser, f"roam.spotId !== {first!r} && !roam.fog")
        assert js(browser, "roam.built") == 2       # the child placed beside its parent
        # ...and the walk goes on north, past where the old spot's edge held it.
        wait(browser, f"roam.world.z < {-edge - 0.3}")
    finally:
        key_up(browser, "w")
    wait(browser, "Number(getComputedStyle(document.getElementById('fog')).opacity) < 0.05")
    assert max(stalls(js(browser, "roam.samples"), walk_speed), default=0) <= 100
    assert js(browser, "window.roamErrors") == []


FOG_TEXT = "document.getElementById('fog').textContent"


def test_fog_shows_progress(server, browser):
    url, svc, painter = server
    painter.gate.clear()                 # every step blocks: the pre-dream ahead paints, and says how far it got
    painter.report = (3, 8)
    start(browser, url)
    first = js(browser, "roam.spotId")
    assert "painting" not in js(browser, FOG_TEXT)
    browser.execute_script("roamDev.look(0, 0.18)")
    key_down(browser, "w")
    try:
        assert painter.entered.wait(10)
        wait(browser, "roam.fog && roam.braking", timeout=10)
        wait(browser, f"{FOG_TEXT}.includes('painting… 3/8')", timeout=5)    # polled every second
        assert "dreaming the way ahead" in js(browser, FOG_TEXT)
        painter.gate.set()
        wait(browser, f"roam.spotId !== {first!r} && !roam.fog")
    finally:
        key_up(browser, "w")
    wait(browser, f"!{FOG_TEXT}.includes('painting')", timeout=5)   # cleared for the next wait
    assert js(browser, "window.roamErrors") == []


def test_first_spot_shows_progress_under_the_intro(server, browser):
    url, _, painter = server
    painter.pano_delay, painter.report = 4.0, (5, 8)
    open_game(browser, url)
    submit(browser, "a quiet forest")
    wait(browser, "document.getElementById('start-progress').textContent === 'painting… 5/8'", timeout=4)
    wait(browser, "roam.started && roam.spotId !== null")
    wait(browser, "document.getElementById('start-progress').textContent === ''", timeout=3)
    assert js(browser, "window.roamErrors") == []


def test_failed_next_keeps_current_spot(server, browser):
    url, svc, painter = server
    painter.fail_steps = True            # every pre-dream and demand build is a GPU out-of-memory
    start(browser, url)
    first = js(browser, "roam.spotId")
    browser.execute_script("roamDev.look(90, 0.18)")
    n = len(svc.calls)
    key_down(browser, "w")
    try:
        wait(browser, TOAST)
        assert "out of memory" in browser.find_element("id", "toast").text
        assert "try another direction" in browser.find_element("id", "toast").text
        wait(browser, "roam.stuck && roamDev.hero.speed < 0.01", timeout=10)
        time.sleep(0.6)                  # still holding W at the rim: no retry storm, no treadmill, no fog
        assert len(svc.since(n, "next")) == 1
        assert js(browser, "roamDev.hero.walk.getEffectiveWeight()") < 0.1
        assert js(browser, "roamDev.hero.speed") < 0.05
        assert not js(browser, "roam.fog")
        assert radius(browser) <= rim(browser) + 1e-6
    finally:
        key_up(browser, "w")
    assert js(browser, "roam.spotId") == first and js(browser, "roam.built") == 1
    assert not js(browser, "roam.stepping")
    r = radius(browser)
    key_down(browser, "s")               # still playable: walk back in, which opens the way again
    try:
        wait(browser, "!roam.stuck", timeout=20)
    finally:
        key_up(browser, "s")
    assert radius(browser) < r - 0.6
    key_down(browser, "w")
    try:
        wait_for(lambda: len(svc.since(n, "next")) == 2, timeout=20)
    finally:
        key_up(browser, "w")
    assert js(browser, "window.roamErrors") == []


def bearing(a, b):
    """World bearing (degrees) from point a to point b, both {x, z}."""
    return math.degrees(math.atan2(b["x"] - a["x"], -(b["z"] - a["z"]))) % 360


def step_into_a_child(d, svc):
    """Walk north from the first spot until the server steps into the next one; returns (first, second) ids."""
    first = js(d, "roam.spotId")
    d.execute_script("roamDev.look(0, 0.18)")
    key_down(d, "w")
    try:
        wait(d, f"roam.spotId !== {first!r} && roam.inServerSpot")
    finally:
        key_up(d, "w")
    return first, js(d, "roam.spotId")


def test_walking_back_then_out_steps_from_the_spot_the_player_is_in(server, browser):
    """Back in the previous spot (past the bisector), the server is told (/enter), and the next /next is asked from
    that spot: its exit heading is the bearing from that spot's origin to where the player walks out."""
    url, svc, _ = server
    start(browser, url)
    first, second = step_into_a_child(browser, svc)
    n = len(svc.calls)
    browser.execute_script("roamDev.look(180, 0.18)")
    key_down(browser, "w")
    try:
        wait(browser, f"roam.spotId === {first!r} && !roam.entering")    # walked back in, and the server knows
    finally:
        key_up(browser, "w")
    assert ("enter", first) in svc.since(n)
    browser.execute_script("roamDev.look(90, 0.18)")
    key_down(browser, "w")
    try:
        wait(browser, f"roam.steps.length === 2 && roam.spotId !== {first!r} && !roam.stepping")
    finally:
        key_up(browser, "w")
    step = js(browser, "roam.steps")[1]
    assert step["from"] == first and step["inServerSpot"], step
    assert js(browser, "roam.parentSpotId") == first
    nexts = [h for h, _ in svc.since(n, "next")]
    assert nexts == [pytest.approx(step["exit"] % 360)]
    assert ang(step["exit"], bearing(step["origin"], step["player"])) <= 10, step
    assert 45 < step["exit"] < 135, step           # east of the first spot
    assert js(browser, "window.roamErrors") == []


def test_walking_back_through_the_parent_and_out_its_far_side(server, browser):
    """Along the child→parent line and on out of the parent: one switch back, steady movement, no re-root
    ping-pong, and the next spot asked from the parent."""
    url, svc, _ = server
    start(browser, url)
    first, second = step_into_a_child(browser, svc)
    browser.execute_script("roamDev.look(180, 0.18)")
    before, z0 = js(browser, "roam.switches"), js(browser, "roam.world.z")
    key_down(browser, "w")
    try:
        wait(browser, f"roam.world.z > {z0 + 1.5}", timeout=30)        # kept walking south (by distance, not clock)
        assert js(browser, "roam.switches") - before <= 2, js(browser, "JSON.stringify(roam)")
        wait(browser, "roam.steps.length === 2")
    finally:
        key_up(browser, "w")
    step = js(browser, "roam.steps")[1]
    assert step["from"] == first and step["inServerSpot"], step
    assert ang(step["exit"], 180) < 20, step
    assert js(browser, "window.roamErrors") == []


def walk(d, bearing, until, timeout=60):
    """Hold W looking along `bearing` until the JS condition holds."""
    d.execute_script(f"roamDev.look({bearing}, 0.18)")
    key_down(d, "w")
    try:
        wait(d, until, timeout=timeout)
    finally:
        key_up(d, "w")


def test_back_into_a_spot_after_exploring_three_ways_from_it(server, browser):
    """Out and back from the first spot three times (north, east, west): walking back from the third child still
    steps the server into the first spot, and there is no switching back and forth."""
    url, svc, _ = server
    start(browser, url)
    first = js(browser, "roam.spotId")
    here = f"roam.spotId === {first!r} && !roam.entering && roam.inServerSpot"
    away = f"roam.spotId !== {first!r} && roam.inServerSpot && !roam.stepping"
    walk(browser, 0, away)           # child 1, north
    walk(browser, 180, here)         # back
    walk(browser, 90, away)          # child 2, east
    walk(browser, 270, here)         # back...
    walk(browser, 270, away)         # ...and on out the far side: child 3, west
    third = js(browser, "roam.spotId")
    n = len(svc.calls)
    browser.execute_script("roamDev.look(90, 0.18)")
    before, x0 = js(browser, "roam.switches"), js(browser, "roam.world.x")
    key_down(browser, "w")
    try:
        wait(browser, here, timeout=20)          # back in the first spot, the server with it
        # Kept walking east (by distance: a loaded machine runs the game slower than the clock), with at most the
        # switch back plus one on into a new spot on the way.
        wait(browser, f"roam.world.x > {x0 + 1.5}", timeout=30)
        assert js(browser, "roam.switches") - before <= 2, js(browser, "JSON.stringify(roam)")
    finally:
        key_up(browser, "w")
    assert ("enter", first) in svc.since(n)
    assert js(browser, "roam.spotId") != third
    assert "faded" not in (js(browser, "document.getElementById('toast').textContent") or "")
    assert js(browser, "window.roamErrors") == []


def to_the_middle(d, near=0.4):
    """Walk to the current spot's centre (re-aiming as it goes)."""
    for _ in range(60):
        pos = js(d, "roam.pos")
        if math.hypot(pos["x"], pos["z"]) < near:
            return
        d.execute_script(f"roamDev.look({bearing(pos, {'x': 0, 'z': 0})}, 0.18)")
        hold(d, "w", 0.15 if math.hypot(pos["x"], pos["z"]) < 1 else 0.4)
        wait(d, "roamDev.hero.speed < 0.01")
    raise AssertionError(f"never reached the middle: {js(d, 'roam.pos')}")


def test_back_in_a_spot_its_oldest_child_is_walked_into_again(server, browser):
    """8b: from the first spot out three ways (north, east, west) and back; then north again walks into the very
    first child (the viewer and the server both still hold it): /enter succeeds, no new /next, no 'faded' toast, no
    switching back and forth, and it stays."""
    url, svc, _ = server
    start(browser, url)
    first = js(browser, "roam.spotId")
    here = f"roam.spotId === {first!r} && !roam.entering && roam.inServerSpot"
    away = f"roam.spotId !== {first!r} && roam.inServerSpot && !roam.stepping"
    walk(browser, 0, away)
    north = js(browser, "roam.spotId")
    walk(browser, 180, here)
    walk(browser, 90, away)
    walk(browser, 270, here)
    walk(browser, 270, away)
    walk(browser, 90, here)
    to_the_middle(browser)                           # where a fresh walk north would start from
    assert js(browser, "roam.spotId") == first
    assert north in js(browser, "roam.spotIds")
    n, before = len(svc.calls), js(browser, "roam.switches")
    walk(browser, 0, f"roam.spotId === {north!r} && !roam.entering && roam.inServerSpot", timeout=30)
    time.sleep(1.0)
    assert js(browser, "roam.spotId") == north and js(browser, "roam.inServerSpot")
    assert [k for k, _ in svc.since(n) if k in ("next", "enter")] == ["enter"]
    assert svc.since(n, "enter")[0][0] == north
    assert js(browser, "roam.switches") - before == 1
    assert "faded" not in (js(browser, "document.getElementById('toast').textContent") or "")
    assert js(browser, "window.roamErrors") == []


def test_a_refused_enter_drops_the_spot_and_does_not_loop(server, browser):
    """The server no longer holds the spot walked back into (unknown_spot): the viewer says so, drops it, re-roots
    on the server's spot, asks only once, and the next step goes out from that spot."""
    url, svc, painter = server
    start(browser, url)
    first, second = step_into_a_child(browser, svc)
    svc.refuse_enter = True
    painter.gate.clear()                 # no further step can land before the checks below
    n = len(svc.calls)
    browser.execute_script("roamDev.look(180, 0.18)")
    key_down(browser, "w")
    try:
        wait(browser, f"roam.entering || !roam.spotIds.includes({first!r})", timeout=30)   # switched back in
    finally:
        key_up(browser, "w")                         # stop there: no step may go out before the checks below
    wait(browser, TOAST)
    assert "faded" in browser.find_element("id", "toast").text
    wait(browser, f"!roam.spotIds.includes({first!r}) && !roam.entering")
    wait(browser, "roamDev.hero.speed < 0.01")
    # the client's current spot is the second one, and it is the server's spot
    assert js(browser, "roamDev.world.current.meta.spot_id") == second
    assert js(browser, "roam.spotId") == second and js(browser, "roam.inServerSpot")
    steps = js(browser, "roam.steps")
    assert len(steps) == 1 or (len(steps) == 2 and js(browser, "roam.stepping")), steps   # none has landed
    painter.gate.set()
    walk(browser, 180, "roam.steps.length >= 2 && !roam.stepping", timeout=30)   # on that way: a new spot
    step = js(browser, "roam.steps")[1]
    assert step["from"] == second and step["inServerSpot"], step
    assert [a for a, _ in svc.since(n, "enter")] == [first]        # asked once, no retry storm
    assert first not in js(browser, "roam.spotIds")
    assert js(browser, "window.roamErrors") == []


def test_a_step_that_lands_while_braking_at_the_far_rim_is_not_entered(server, browser):
    """A step asked for northward lands while the player has turned round and is waiting at the south rim: the
    player stays put (not moved into the spot ahead), no arrival toast, and the spot is walked into later."""
    url, svc, painter = server
    painter.gate.clear()                 # every step blocks until released
    start(browser, url)
    first = js(browser, "roam.spotId")
    walk(browser, 0, "roam.stepping")    # the step north is out
    browser.execute_script("roamDev.look(180, 0.18)")
    key_down(browser, "w")               # held throughout: braking at the south rim when the step lands
    try:
        wait(browser, "roam.braking && roamDev.hero.speed < 0.02", timeout=30)
        pos = js(browser, "roam.pos")
        assert pos["z"] > 0 and radius(browser) >= 0.9 * rim(browser)
        browser.execute_script("document.getElementById('toast').hidden = true")
        painter.gate.set()                # the north step lands...
        time.sleep(0.05)
        painter.gate.clear()              # ...and nothing after it (the walk back's own step would)
        wait(browser, "roam.spotIds.length === 2")
        ahead = next(x for x in js(browser, "roam.spotIds") if x != first)    # the spot dreamed north
        state = js(browser, "JSON.stringify({cur: roamDev.world.current.meta.spot_id, pos: roam.pos, "
                            "shift: roamDev.world.landShift, toast: document.getElementById('toast').hidden ? '' : "
                            "document.getElementById('toast').textContent})")
    finally:
        key_up(browser, "w")
    import json
    state = json.loads(state)
    assert state["cur"] == first, state                     # not entered from this rim
    assert math.hypot(state["pos"]["x"] - pos["x"], state["pos"]["z"] - pos["z"]) < 0.05, (pos, state)
    assert "Arrived" not in state["toast"], state
    painter.gate.set()
    wait(browser, "roamDev.hero.speed < 0.01 && !roam.entering && !roam.stepping")
    assert ahead in js(browser, "roam.spotIds")
    to_the_middle(browser)
    walk(browser, 0, f"roamDev.world.current.meta.spot_id === {ahead!r} && roam.inServerSpot && !roam.entering",
         timeout=40)                      # still there to walk into, the way it was dreamed
    assert js(browser, "window.roamErrors") == []


def test_creeping_up_to_the_rim_and_turning_back_commits_nothing(server, browser):
    """The commit point depends on speed: creeping in short taps to well past where a walk would commit, then
    turning back, sends no /next."""
    url, svc, _ = server
    start(browser, url)
    n = len(svc.calls)
    browser.execute_script("roamDev.look(0, 0.18)")
    edge = rim(browser)
    hold(browser, "w", 1.0)
    # Walking at full speed commits about 1.15 short of the rim (coast + LEAD); creep to within 0.7 of it.
    for _ in range(120):
        if radius(browser) >= edge - 0.7 or js(browser, "roam.steps.length"):
            break
        hold(browser, "w", 0.1 if radius(browser) > edge - 1.8 else 0.6)
        wait(browser, "roamDev.hero.speed < 0.01")
    assert js(browser, "roam.steps") == [] and radius(browser) >= edge - 0.7
    browser.execute_script("roamDev.look(180, 0.18)")
    hold(browser, "w", 1.0)
    assert radius(browser) < edge - 1.2
    assert svc.since(n, "next") == [] and js(browser, "roam.steps") == []


def back_until_open(d):
    """Walk backwards (S) until the way out opens again (inside RETRY_FROM of the rim)."""
    key_down(d, "s")
    try:
        wait(d, "!roam.stuck", timeout=20)
    finally:
        key_up(d, "s")


def fail_a_build_walking_north(d, svc):
    """Walk north with the next spot's panorama hidden: /next succeeds, its spot can't build."""
    svc.hide_next_image = True
    d.execute_script("roamDev.look(0, 0.18)")
    key_down(d, "w")
    try:
        wait(d, TOAST)
        assert "didn't load" in d.find_element("id", "toast").text
        assert "step back and try again" in d.find_element("id", "toast").text
    finally:
        key_up(d, "w")


def test_spot_that_failed_to_build_is_built_again_without_a_new_next(server, browser):
    url, svc, _ = server
    start(browser, url)
    first = js(browser, "roam.spotId")
    n = len(svc.calls)
    fail_a_build_walking_north(browser, svc)
    assert js(browser, "roam.spotId") == first
    aside, original = svc.hidden
    aside.rename(original)
    back_until_open(browser)
    # Out to the east the way stays closed: the server's next spot lies north.
    browser.execute_script("document.getElementById('toast').hidden = true; roamDev.look(90, 0.18)")
    key_down(browser, "w")
    try:
        wait(browser, f"{TOAST} && roam.braking", timeout=20)
        assert "step back and try again" in browser.find_element("id", "toast").text
        wait(browser, "roamDev.hero.speed < 0.01")
        assert not js(browser, "roam.fog") and not js(browser, "roam.stepping")
    finally:
        key_up(browser, "w")
    assert len(js(browser, "roam.steps")) == 1
    hold(browser, "s", 2.5)                           # back west, toward the middle
    browser.execute_script("roamDev.look(0, 0.18)")
    key_down(browser, "w")
    try:
        wait(browser, f"roam.spotId !== {first!r}")
    finally:
        key_up(browser, "w")
    assert len(svc.since(n, "next")) == 1              # the spot the server had moved to, built this time
    assert js(browser, "roam.steps.map(s => s.rebuild)") == [False, True]
    assert js(browser, "roam.spotId") == original.stem
    assert js(browser, "window.roamErrors") == []


def test_spot_that_keeps_failing_to_build_is_given_up(server, browser):
    url, svc, _ = server
    start(browser, url)
    first = js(browser, "roam.spotId")
    n = len(svc.calls)
    fail_a_build_walking_north(browser, svc)           # its panorama never comes back
    back_until_open(browser)
    browser.execute_script("roamDev.look(0, 0.18)")
    key_down(browser, "w")
    try:
        wait_for(lambda: ("enter", first) in svc.since(n), timeout=30)   # given up: the server steps back
        wait(browser, f"roam.spotId !== {first!r}")          # and a fresh /next goes out and lands
    finally:
        key_up(browser, "w")
    assert [k for k, _ in svc.since(n) if k in ("next", "enter")][:3] == ["next", "enter", "next"]
    assert js(browser, "roam.steps.map(s => s.rebuild)")[:3] == [False, True, False]
    assert js(browser, "window.roamErrors") == []


def test_chip_toggled_during_start_is_sent_after_it(server, browser):
    url, svc, painter = server
    painter.pano_delay = 1.5
    open_game(browser, url)
    n = len(svc.calls)
    submit(browser, "a quiet forest")
    wait(browser, "document.getElementById('place').disabled")
    browser.find_element("css selector", ".chip[data-mood='snow']").click()
    wait(browser, "roam.started && roam.spotId !== null")
    wait_for(lambda: len(svc.since(n)) == 2)
    assert svc.since(n) == [("start", ("a quiet forest", [])), ("moods", ["snow"])]


def test_signposts_and_compass_show_the_neighbours(server, browser):
    url, _, _ = server
    start(browser, url)
    names = js(browser, "roam.neighbours.map(n => n.name)")
    assert len(names) >= 2
    assert sorted(js(browser, "roam.signposts")) == sorted(names)
    shown = js(browser, "[...document.querySelectorAll('#compass .place')].map(e => e.dataset.name)")
    assert sorted(shown) == sorted(names)
    assert all(t in js(browser, "document.getElementById('compass').textContent") for t in names)
    # The signpost stands near the spot centre, in view from the start.
    x, y = js(browser, "roamDev.signpostOnScreen()")
    assert 0 < x < 1 and 0 < y < 1, (x, y)
    # The compass follows the camera: the neighbour straight ahead sits at its centre.
    b = js(browser, "roam.neighbours[0].bearing")
    browser.execute_script(f"roamDev.look({b}, 0.18)")
    wait(browser, f"""(() => {{ const c = document.getElementById('compass').getBoundingClientRect();
        const e = document.querySelector('#compass .place[data-name="{names[0]}"]').getBoundingClientRect();
        return Math.abs((e.left + e.right) / 2 - (c.left + c.right) / 2) < 4; }})()""")


def test_heading_hint_is_throttled(server, browser):
    url, svc, _ = server
    start(browser, url)
    n = len(svc.calls)
    time.sleep(0.8)                      # standing still: nothing to say
    assert svc.since(n, "heading") == []
    # Walk in tight circles: the heading changes fast, the hints still come at most every 300 ms, > 20° apart.
    browser.execute_script("roamDev.spin = setInterval(() => roamDev.look(roam.camBearing + 15), 50)")
    hold(browser, "w", 2.5)
    browser.execute_script("clearInterval(roamDev.spin)")
    sent = js(browser, "roam.hints")
    assert len(sent) >= 4, sent
    for a, b in zip(sent, sent[1:]):
        assert b["t"] - a["t"] >= 299, sent
        assert ang(a["heading"], b["heading"]) > 20, sent
    wait_for(lambda: len(svc.since(n, "heading")) == len(sent))
    assert [round(h, 3) for h, _ in svc.since(n, "heading")] == [round(s["heading"] % 360, 3) for s in sent]


def test_text_bar_sends_theme_and_chips_send_moods(server, browser):
    url, svc, _ = server
    start(browser, url)
    n = len(svc.calls)
    submit(browser, "a stormy coast")
    wait(browser, TOAST)
    assert "a stormy coast" in browser.find_element("id", "toast").text
    browser.find_element("css selector", ".chip[data-mood='rain']").click()
    wait_for(lambda: ("moods", ["rain"]) in svc.since(n))
    assert svc.since(n) == [("theme", "a stormy coast"), ("moods", ["rain"])]


def test_walking_toward_a_place_arrives(server, browser):
    url, _, _ = server
    start(browser, url)
    target = js(browser, "roam.neighbours[0]")
    browser.execute_script(f"roamDev.look({target['bearing']}, 0.18)")
    key_down(browser, "w")
    try:
        # the toast comes when the player is in the spot that arrived (not when the server moves there)
        wait(browser, f"roam.place === {target['name']!r} && roam.inServerSpot", timeout=120)
    finally:
        key_up(browser, "w")
    wait(browser, TOAST)
    assert browser.find_element("id", "toast").text == f"Arrived at {target['name']}"
    names = js(browser, "roam.neighbours.map(n => n.name)")
    assert "a quiet forest" in names                      # where you came from, behind you
    shown = js(browser, "[...document.querySelectorAll('#compass .place')].map(e => e.dataset.name)")
    assert sorted(shown) == sorted(names)
    assert js(browser, "document.querySelector('#compass .came-from').hidden") is False
    assert js(browser, "window.roamErrors") == []


def phone(d, url, w=390, h=844):
    """The game in a w×h viewport. Firefox won't make a window narrower than 500 px, so it runs in an iframe of
    that size, which is the viewport its media queries and layout see."""
    d.set_window_size(600, h + 200)
    d.get(f"data:text/html,<body style='margin:0'><iframe id='phone' src='{url}' width='{w}' height='{h}' "
          "style='border:0'></iframe></body>")
    d.switch_to.frame(d.find_element("id", "phone"))
    wait(d, "window.roam && roam.ready")


def test_phone_stick_walks(server, browser):
    from selenium.webdriver.common.action_chains import ActionChains

    url, _, _ = server
    phone(browser, url)
    submit(browser, "a quiet forest")
    wait(browser, "roam.started && roam.spotId !== null")
    stick = browser.find_element("id", "stick")
    assert stick.is_displayed()
    r = js(browser, "document.getElementById('stick').getBoundingClientRect().toJSON()")
    assert r["right"] < 390 / 2 and r["top"] > 844 / 2           # bottom left
    ActionChains(browser).move_to_element(stick).click_and_hold().move_by_offset(0, -40).pause(0.6) \
        .release().perform()
    assert js(browser, "roam.pos.z") < -0.1                        # pushed up: walked forward (north)
    yaw = js(browser, "roam.camYaw")
    canvas = browser.find_element("css selector", "#stage canvas")
    ActionChains(browser).move_to_element(canvas).click_and_hold().move_by_offset(-80, 0).release().perform()
    assert js(browser, "roam.camYaw") > yaw + 0.2
    assert js(browser, "window.roamErrors") == []
