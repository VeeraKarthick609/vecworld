"""Browser tests for the play screen, against a server running fake models (headless Firefox)."""
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from conftest import FakeEmbedder, FakeGenerator, OomGenerator, make_space
from vecworld.__main__ import WEB_DIR
from vecworld.api.app import create_app
from vecworld.game.service import GameService
from vecworld.scenes.hotspots import Hotspot

pytestmark = pytest.mark.ui


class SwitchableGenerator(FakeGenerator):
    """Works until .fail is set, then behaves like a GPU out of memory. .delay slows every image."""

    fail = False
    delay = 0.0

    def fresh(self, prompt, seed):
        time.sleep(self.delay)
        return OomGenerator.fresh(self, prompt, seed) if self.fail else super().fresh(prompt, seed)

    def evolve(self, prev, prompt, seed):
        time.sleep(self.delay)
        return OomGenerator.fresh(self, prompt, seed) if self.fail else super().evolve(prev, prompt, seed)


class LayoutFinder:
    """Centre in the middle, first neighbour small on the left, second small on the right,
    third a big diffuse box underneath both (it must not cover them), the rest small at the top."""

    def find(self, img, concepts):
        boxes = [(0.4, 0.4, 0.2, 0.2), (0.05, 0.6, 0.15, 0.15), (0.8, 0.6, 0.15, 0.15), (0.0, 0.3, 0.9, 0.5)]
        boxes += [(0.1 * i, 0.02, 0.08, 0.08) for i in range(len(concepts))]
        return [Hotspot(c, *b) for c, b in zip(concepts, boxes)]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    gen = SwitchableGenerator()
    svc = GameService(make_space(), np.zeros(3, np.float32), FakeEmbedder(), gen, LayoutFinder(),
                      tmp_path_factory.mktemp("scenes"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(svc, WEB_DIR)

    @app.middleware("http")
    async def record(request, call_next):
        if request.url.path.startswith("/api/"):
            gen.requests.append(request.url.path.rsplit("/", 1)[-1])
        return await call_next(request)

    gen.requests = []
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", gen
    srv.should_exit = True


@pytest.fixture
def browser():
    from selenium import webdriver
    from selenium.webdriver.firefox.service import Service

    opts = webdriver.FirefoxOptions()
    opts.add_argument("-headless")
    d = webdriver.Firefox(options=opts, service=Service("/snap/bin/geckodriver"))
    d.set_window_size(1000, 800)
    yield d
    d.quit()


def wait(d, js, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if d.execute_script(f"return {js}"):
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for: {js}")


def jump(d, url, text):
    from selenium.webdriver.common.keys import Keys

    if d.current_url != url + "/":
        d.get(url)
    box = d.find_element("id", "jump-text")
    box.clear()
    box.send_keys(text, Keys.ENTER)


SCENE_VISIBLE = "[...document.querySelectorAll('.scene.shown')].some(e => e.src && getComputedStyle(e).opacity > 0.9)"
IDLE = "!document.body.classList.contains('busy')"
TOAST = "!document.getElementById('toast').hidden"


@pytest.fixture(autouse=True)
def healthy(server):
    _, gen = server
    gen.fail, gen.delay = False, 0.0
    yield


def press(d, key):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).send_keys(key).perform()


def landed(d, url, text="ocean"):
    jump(d, url, text)
    wait(d, f"{IDLE} && document.querySelectorAll('.hotspot').length > 0")


def centre(d):
    return d.execute_script("return vecworld.scene.centre")


def test_failed_jump_keeps_scene_visible(server, browser):
    url, gen = server
    gen.fail = False
    jump(browser, url, "ocean")
    wait(browser, f"{IDLE} && document.querySelectorAll('.hotspot').length > 0")
    gen.fail = True
    jump(browser, url, "sea")
    wait(browser, f"{IDLE} && {TOAST}")
    time.sleep(0.5)
    assert browser.execute_script(f"return {SCENE_VISIBLE}")
    assert browser.execute_script("return document.querySelectorAll('.hotspot').length") > 0


def test_failed_step_restores_hotspots(server, browser):
    url, gen = server
    gen.fail = False
    jump(browser, url, "ocean")
    wait(browser, f"{IDLE} && document.querySelectorAll('.hotspot').length > 0")
    before = browser.execute_script("return document.querySelectorAll('.hotspot').length")
    gen.fail = True
    browser.execute_script("document.querySelectorAll('.hotspot')[1].click()")
    wait(browser, f"{IDLE} && {TOAST}")
    time.sleep(0.5)
    assert browser.execute_script("return document.querySelectorAll('.hotspot').length") == before
    assert browser.execute_script(f"return {SCENE_VISIBLE}")


def test_small_hotspot_is_not_covered_by_a_big_one(server, browser):
    url, gen = server
    gen.fail = False
    jump(browser, url, "ocean")
    wait(browser, f"{IDLE} && document.querySelectorAll('.hotspot').length > 1")
    covered = browser.execute_script("""
        return [...document.querySelectorAll('.hotspot')].filter(h => {
          const r = h.getBoundingClientRect();
          const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
          return !(hit === h || h.contains(hit));
        }).map(h => h.getBoundingClientRect().width < 0.5 * innerWidth);
    """)
    # Every smaller box must be reachable; only the full-frame one may be covered at its centre.
    assert True not in covered


def test_labels_are_visible_without_hovering(server, browser):
    url, _ = server
    landed(browser, url)
    opacities = browser.execute_script(
        "return [...document.querySelectorAll('.hotspot span')].map(s => +getComputedStyle(s).opacity)")
    assert opacities and min(opacities) > 0.9


def test_w_walks_forward_and_s_comes_back(server, browser):
    from selenium.webdriver.common.keys import Keys

    url, gen = server
    landed(browser, url)
    first = browser.execute_script("return vecworld.scene.scene_id")
    press(browser, "w")
    wait(browser, f"{IDLE} && vecworld.scene.scene_id !== '{first}'")
    assert gen.requests[-1] == "forward" and centre(browser) != "ocean"
    press(browser, Keys.ARROW_DOWN)
    wait(browser, f"{IDLE} && vecworld.scene.scene_id === '{first}'")
    assert gen.requests[-1] == "back"
    wait(browser, SCENE_VISIBLE, timeout=2)
    press(browser, Keys.ARROW_UP)
    wait(browser, f"{IDLE} && vecworld.scene.scene_id !== '{first}'")
    assert gen.requests[-1] == "forward"


def test_a_and_d_step_toward_the_left_and_right_objects(server, browser):
    from selenium.webdriver.common.keys import Keys

    url, gen = server
    landed(browser, url)
    left, right = browser.execute_script(
        "const hs = vecworld.scene.hotspots; return [hs[1].concept, hs[2].concept]")
    press(browser, "a")
    wait(browser, f"{IDLE} && vecworld.scene.centre === '{left}'")
    assert gen.requests[-1] == "step"
    landed(browser, url)
    press(browser, Keys.ARROW_RIGHT)
    wait(browser, f"{IDLE} && vecworld.scene.centre === '{right}'")
    assert gen.requests[-1] == "step"


def test_keys_ignored_while_typing_and_before_the_first_jump(server, browser):
    url, gen = server
    browser.get(url)
    browser.find_element("tag name", "body").click()
    n = len(gen.requests)
    press(browser, "w")
    time.sleep(0.5)
    assert gen.requests[n:] in ([], ["session"])          # nothing walks before you've landed
    landed(browser, url)
    box = browser.find_element("id", "jump-text")
    box.click()
    n = len(gen.requests)
    box.send_keys("wasd")
    time.sleep(0.5)
    assert gen.requests[n:] == [] and box.get_attribute("value") == "wasd"


def test_a_key_while_a_move_is_in_flight_sends_nothing(server, browser):
    url, gen = server
    landed(browser, url)
    gen.delay = 1.0
    n = len(gen.requests)
    press(browser, "w")
    time.sleep(0.2)
    press(browser, "w")
    press(browser, "d")
    wait(browser, IDLE)
    assert gen.requests[n:] == ["forward"]


def test_failed_forward_restores_scene_and_hotspots(server, browser):
    url, gen = server
    landed(browser, url)
    before = browser.execute_script("return document.querySelectorAll('.hotspot').length")
    gen.fail = True
    press(browser, "w")
    wait(browser, f"{IDLE} && {TOAST}")
    time.sleep(0.5)
    assert browser.execute_script("return document.querySelectorAll('.hotspot').length") == before
    assert browser.execute_script(f"return {SCENE_VISIBLE}")
