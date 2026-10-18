"""Browser tests for the Dream Walk viewer, against a server running fake models (headless Firefox)."""
import socket
import threading
import time

import pytest
import uvicorn
from conftest import FakeDreamer, FakePainter

from vecworld.__main__ import WEB_DIR
from vecworld.api.app import create_app
from vecworld.config import MOODS
from vecworld.dream.labels import Label
from vecworld.dream.service import DreamService

pytestmark = pytest.mark.ui

SEAM_TEXT = "a lantern on the seam"


class UiPainter(FakePainter):
    """Solid panoramas. .fail_steps makes every step a GPU out-of-memory; .delay slows every step."""

    fail_steps = False
    delay = 0.0

    def step(self, prev, base, focus, focus_at, seed, **kw):
        time.sleep(self.delay)
        if self.fail_steps:
            self.fail_next = True
        return super().step(prev, base, focus, focus_at, seed, **kw)


class UiLabels:
    """Every wanted label where it was asked for, plus one just left of the 360° seam (u = 0.99)."""

    def __init__(self):
        self._n = 0
        self.extra: list[tuple[str, float, float, float, float]] = []   # (text, u, y, w, h) added every step

    def _id(self):
        self._n += 1
        return f"L{self._n}"

    def find(self, img, wanted):
        out = [Label(self._id(), text, u, 0.4, 0.06, 0.2) for text, u in wanted if text != SEAM_TEXT]
        out += [Label(self._id(), *x) for x in self.extra if x[0] not in {t for t, _ in wanted}]
        return out + [Label(self._id(), SEAM_TEXT, 0.99, 0.4, 0.04, 0.2)]


class RecordingService(DreamService):
    """Records every call the viewer makes as (name, argument)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[tuple[str, object]] = []

    def start(self, sid, theme, moods):
        self.calls.append(("start", (theme, list(moods))))
        return super().start(sid, theme, moods)

    def walk(self, sid, yaw):
        self.calls.append(("walk", yaw))
        return super().walk(sid, yaw)

    def back(self, sid, yaw):
        self.calls.append(("back", yaw))
        return super().back(sid, yaw)

    def approach(self, sid, label_id):
        self.calls.append(("approach", label_id))
        return super().approach(sid, label_id)

    def set_theme(self, sid, theme):
        self.calls.append(("theme", theme))
        return super().set_theme(sid, theme)

    def set_moods(self, sid, moods):
        self.calls.append(("moods", list(moods)))
        return super().set_moods(sid, moods)

    def prefetch(self, sid, yaw):
        self.calls.append(("prefetch", yaw))
        return super().prefetch(sid, yaw)

    def names(self, since=0):
        return [n for n, _ in self.calls[since:]]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    painter = UiPainter()
    svc = RecordingService(painter, FakeDreamer(), UiLabels(), tmp_path_factory.mktemp("scenes"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(svc, WEB_DIR), host="127.0.0.1", port=port,
                                        log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", svc, painter
    srv.should_exit = True


@pytest.fixture(autouse=True)
def healthy(server):
    _, _, painter = server
    painter.fail_steps, painter.delay, painter.fail_next = False, 0.0, False
    yield
    server[1].labels.extra = []


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


def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


IDLE = "!document.body.classList.contains('busy')"
TOAST = "!document.getElementById('toast').hidden"
LANDED = f"{IDLE} && window.vecworld && vecworld.view !== null"


def js(d, expr):
    return d.execute_script(f"return {expr}")


def submit(d, text):
    from selenium.webdriver.common.keys import Keys

    box = d.find_element("id", "theme-text")
    box.clear()
    box.send_keys(text, Keys.ENTER)


def land(d, url, theme="a quiet forest"):
    d.get(url)
    started(d, theme)


def started(d, theme="a quiet forest"):
    submit(d, theme)
    wait(d, LANDED)


def phone(d, url, w=390, h=844):
    """Open the viewer in a w×h viewport. Firefox won't make a window narrower than 500 px, so it runs
    in an iframe of that size, which is the viewport its media queries and layout see."""
    d.set_window_size(600, h + 200)
    d.get(f"data:text/html,<body style='margin:0'><iframe id='phone' src='{url}' width='{w}' height='{h}' "
          "style='border:0'></iframe></body>")
    d.switch_to.frame(d.find_element("id", "phone"))
    wait(d, "document.readyState === 'complete' && !!document.getElementById('theme-text')")


def press(d, key):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).send_keys(key).perform()


def hold(d, key, seconds):
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(d).key_down(key).pause(seconds).key_up(key).perform()


def look(d, degrees):
    """Turn by exactly `degrees` with a mouse drag (dragging left turns right)."""
    from selenium.webdriver.common.action_chains import ActionChains

    w, h = js(d, "[innerWidth, innerHeight]")
    fov = min(90, 360 * 0.25 * w / h)          # the fake panoramas are 4:1
    dx = round(degrees * w / fov)
    stage = d.find_element("id", "stage")
    start = max(-w // 2 + 20, min(w // 2 - 20, dx // 2))
    ActionChains(d).move_to_element_with_offset(stage, start, 0).click_and_hold() \
        .move_by_offset(-dx, 0).release().perform()


def session_of(svc):
    return max(svc._sessions.values(), key=lambda s: s.last_used)


def moved(d, old_view_id):
    wait(d, f"{IDLE} && vecworld.view.view_id !== '{old_view_id}'")


def view_id(d):
    return js(d, "vecworld.view.view_id")


def visible_labels(d):
    return js(d, """[...document.querySelectorAll('.label')].filter(e => {
        const r = e.getBoundingClientRect();
        return getComputedStyle(e).display !== 'none' && r.width > 0 && r.right > 0 && r.left < innerWidth;
    }).map(e => e.textContent)""")


def yaw_near(a, b, tol=1.0):
    return abs((a - b + 180) % 360 - 180) < tol


def walked(d, svc):
    """Land and take one step, so there are discoveries to look at."""
    first = view_id(d)
    press(d, "w")
    moved(d, first)
    wait(d, "vecworld.labels.length > 0")


def test_start_shows_panorama_and_chips(server, browser):
    url, svc, _ = server
    browser.get(url)
    chips = js(browser, "[...document.querySelectorAll('.chip')].map(c => c.dataset.mood)")
    assert chips == list(MOODS)
    assert js(browser, "vecworld.view") is None
    browser.find_element("css selector", ".chip[data-mood='night']").click()
    assert js(browser, "document.querySelector(\".chip[data-mood='night']\").getAttribute('aria-pressed')") == "true"
    n = len(svc.calls)
    submit(browser, "a quiet forest")
    wait(browser, LANDED)
    assert svc.calls[n] == ("start", ("a quiet forest", ["night"]))
    assert js(browser, "vecworld.view.moods") == ["night"]
    assert js(browser, "vecworld.yaw") == 0
    # The fake panorama is solid navy; the canvas must be showing it.
    wait(browser, """(() => { const c = document.getElementById('view');
        const p = c.getContext('2d').getImageData(c.width / 2, c.height / 2, 1, 1).data;
        return p[2] > 100 && p[0] < 40; })()""")
    assert js(browser, "document.getElementById('intro').hidden")
    # A toggle after starting is sent straight away.
    browser.find_element("css selector", ".chip[data-mood='rain']").click()
    wait_for(lambda: ("moods", ["night", "rain"]) in svc.calls[n:])


def test_turning_changes_yaw_without_requests(server, browser):
    url, svc, _ = server
    land(browser, url)
    n = len(svc.calls)
    hold(browser, "d", 0.5)
    assert js(browser, "vecworld.yaw") > 30
    hold(browser, "a", 0.5)
    assert js(browser, "vecworld.yaw") < 30
    assert not {"walk", "back", "approach"} & set(svc.names(n))


def test_settled_view_prefetches_once(server, browser):
    url, svc, _ = server
    land(browser, url)
    n = len(svc.calls)
    time.sleep(0.8)
    assert svc.names(n) == ["prefetch"]
    assert yaw_near(svc.calls[n][1], 0)
    # Turning well past 45° and settling prefetches again, toward the new direction.
    hold(browser, "d", 0.5)
    yaw = js(browser, "vecworld.yaw")
    wait_for(lambda: svc.names(n) == ["prefetch", "prefetch"])
    assert yaw_near(svc.calls[-1][1], yaw)


def test_w_walks_and_s_turns_around(server, browser):
    from selenium.webdriver.common.keys import Keys

    url, svc, _ = server
    land(browser, url)
    hold(browser, "d", 0.3)
    yaw = js(browser, "vecworld.yaw")
    first = view_id(browser)
    n = len(svc.calls)
    press(browser, "w")
    moved(browser, first)
    walks = [a for name, a in svc.calls[n:] if name == "walk"]
    assert len(walks) == 1 and yaw_near(walks[0], yaw)
    assert js(browser, "vecworld.view.steps") == 1
    # Still facing the way you walked, whether or not a prefetch toward yaw 0 served it.
    assert yaw_near(js(browser, "vecworld.yaw"), yaw - js(browser, "vecworld.view.walked_yaw"))

    hold(browser, Keys.ARROW_RIGHT, 0.3)
    yaw = js(browser, "vecworld.yaw")
    assert yaw > 20
    second = view_id(browser)
    n = len(svc.calls)
    press(browser, "s")
    moved(browser, second)
    backs = [a for name, a in svc.calls[n:] if name == "back"]
    assert len(backs) == 1 and yaw_near(backs[0], yaw)
    # The server centres the new panorama on what was behind you (or near it), and you now face it.
    walked = js(browser, "vecworld.view.walked_yaw")
    assert yaw_near(js(browser, "vecworld.yaw"), yaw + 180 - walked)
    assert js(browser, "Math.abs(vecworld.yaw)") <= 45


def test_walk_on_a_prefetch_keeps_facing_the_same_way(server, browser):
    url, svc, _ = server
    land(browser, url)
    wait_for(lambda: (pf := session_of(svc).prefetch) is not None and pf[2].done())
    look(browser, 30)
    yaw = js(browser, "vecworld.yaw")
    assert yaw_near(yaw, 30, tol=2), yaw
    first = view_id(browser)
    press(browser, "w")
    moved(browser, first)
    assert yaw_near(js(browser, "vecworld.view.walked_yaw"), 0)    # the prefetch toward 0 was used
    assert yaw_near(js(browser, "vecworld.yaw"), yaw)               # and you still face 30° off it


def test_label_click_approaches(server, browser):
    url, svc, _ = server
    land(browser, url)
    walked(browser, svc)
    ahead = js(browser, "vecworld.labels.find(l => Math.abs(l.u - 0.5) < 0.01)")
    assert ahead is not None
    el = browser.find_element("css selector", f".label[data-id='{ahead['id']}']")
    assert el.is_displayed() and el.text == ahead["text"]
    before = view_id(browser)
    n = len(svc.calls)
    el.click()
    moved(browser, before)
    assert ("approach", ahead["id"]) in svc.calls[n:]
    assert yaw_near(js(browser, "vecworld.yaw"), 0)


def test_tall_label_name_stays_on_screen(server, browser):
    url, svc, _ = server
    svc.labels.extra = [("a huge old oak", 0.65, 0.0, 0.25, 1.0)]   # full height, mostly off to the right
    land(browser, url)
    walked(browser, svc)
    wait(browser, "[...document.querySelectorAll('.label span')].some(s => s.textContent === 'a huge old oak')")
    r = js(browser, """(() => { const s = [...document.querySelectorAll('.label span')]
        .find(s => s.textContent === 'a huge old oak'); return s.getBoundingClientRect().toJSON(); })()""")
    w, h = js(browser, "[innerWidth, innerHeight]")
    assert 0 <= r["left"] and r["right"] <= w and 0 <= r["top"] and r["bottom"] <= h, r


def test_label_draws_across_seam(server, browser):
    from selenium.webdriver.common.action_chains import ActionChains

    url, svc, _ = server
    land(browser, url)
    walked(browser, svc)
    assert SEAM_TEXT in js(browser, "vecworld.labels.map(l => l.text)")
    assert SEAM_TEXT not in visible_labels(browser)
    # Drag right three times: 3 × 659 px of a 1000 px wide 90° view turns to yaw ≈ -178,
    # the other side of the panorama's seam from u = 0.99.
    w = js(browser, "innerWidth")
    stage = browser.find_element("id", "stage")
    for _ in range(3):
        ActionChains(browser).move_to_element_with_offset(stage, -w // 2 + 100, 0).click_and_hold() \
            .move_by_offset(659 * w // 1000, 0).release().perform()
    yaw = js(browser, "vecworld.yaw")
    assert yaw_near(yaw, -178, tol=3), yaw
    assert SEAM_TEXT in visible_labels(browser)
    x = js(browser, f"""(() => {{ const e = [...document.querySelectorAll('.label')]
        .find(e => e.textContent === {SEAM_TEXT!r}); const r = e.getBoundingClientRect();
        return (r.left + r.right) / 2 / innerWidth; }})()""")
    assert 0.3 < x < 0.5     # u 0.99 sits 0.0156 of a turn left of centre at yaw -178


def test_failed_walk_keeps_view(server, browser):
    url, svc, painter = server
    land(browser, url)
    walked(browser, svc)
    before = js(browser, "[vecworld.view.view_id, vecworld.labels.map(l => l.id)]")
    # Fail every step from now on, and turn well away from yaw 0 first, so a prefetch built
    # before the failure started can't serve this walk.
    painter.fail_steps = True
    look(browser, 52)            # past the 45° prefetch tolerance, with the label ahead still in view
    assert yaw_near(js(browser, "vecworld.yaw"), 52, tol=2)
    shown = visible_labels(browser)
    assert shown
    n = len(svc.calls)
    press(browser, "w")
    wait(browser, f"{IDLE} && {TOAST}")
    assert "walk" in svc.names(n)
    assert "out of memory" in browser.find_element("id", "toast").text
    assert js(browser, "[vecworld.view.view_id, vecworld.labels.map(l => l.id)]") == before
    assert visible_labels(browser) == shown
    # Still looking around after the failure.
    yaw = js(browser, "vecworld.yaw")
    look(browser, -30)
    assert yaw_near(js(browser, "vecworld.yaw"), yaw - 30, tol=2)


def test_text_bar_drifts_theme_after_start(server, browser):
    url, svc, _ = server
    land(browser, url)
    n = len(svc.calls)
    submit(browser, "a stormy coast")
    wait(browser, TOAST)
    assert "the dream is drifting toward a stormy coast" in browser.find_element("id", "toast").text
    assert ("theme", "a stormy coast") in svc.calls[n:]
    assert "start" not in svc.names(n)


def test_keys_ignored_while_typing(server, browser):
    url, svc, painter = server
    land(browser, url)
    box = browser.find_element("id", "theme-text")
    box.click()
    n = len(svc.calls)
    box.send_keys("wasd")
    hold(browser, "d", 0.3)
    assert box.get_attribute("value") == "wasdd"
    assert js(browser, "vecworld.yaw") == 0
    assert not {"walk", "back"} & set(svc.names(n))

    # And while a move is in flight.
    browser.find_element("id", "stage").click()
    painter.delay = 1.0
    first = view_id(browser)
    n = len(svc.calls)
    press(browser, "w")
    wait(browser, "document.body.classList.contains('busy')")
    press(browser, "w")
    press(browser, "s")
    moved(browser, first)
    assert [x for x in svc.names(n) if x in ("walk", "back")] == ["walk"]


def test_phone_buttons_walk(server, browser):
    url, svc, _ = server
    phone(browser, url)
    started(browser)
    assert js(browser, "[innerWidth, innerHeight]") == [390, 844]
    up = browser.find_element("id", "go-walk")
    down = browser.find_element("id", "go-back")
    assert up.is_displayed() and down.is_displayed()
    r = js(browser, "document.getElementById('go-walk').getBoundingClientRect().toJSON()")
    assert r["left"] > 390 / 2 and r["top"] > 844 / 2      # bottom right
    first = view_id(browser)
    n = len(svc.calls)
    up.click()
    moved(browser, first)
    assert "walk" in svc.names(n)
    second = view_id(browser)
    down.click()
    moved(browser, second)
    assert "back" in svc.names(n)
