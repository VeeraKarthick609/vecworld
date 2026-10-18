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
    """Works until .fail is set, then behaves like a GPU out of memory."""

    fail = False

    def fresh(self, prompt, seed):
        return OomGenerator.fresh(self, prompt, seed) if self.fail else super().fresh(prompt, seed)

    def evolve(self, prev, prompt, seed):
        return OomGenerator.fresh(self, prompt, seed) if self.fail else super().evolve(prev, prompt, seed)


class BigLastFinder:
    """Small boxes first, then a diffuse last concept ("sky") whose box covers the whole scene."""

    def find(self, img, concepts):
        boxes = [Hotspot(c, 0.1 + 0.25 * i, 0.6, 0.15, 0.15) for i, c in enumerate(concepts[:-1])]
        return boxes + [Hotspot(concepts[-1], 0.0, 0.0, 1.0, 1.0)]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    gen = SwitchableGenerator()
    svc = GameService(make_space(), np.zeros(3, np.float32), FakeEmbedder(), gen, BigLastFinder(),
                      tmp_path_factory.mktemp("scenes"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(svc, WEB_DIR), host="127.0.0.1", port=port, log_level="warning"))
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
