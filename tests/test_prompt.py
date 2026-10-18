from PIL import Image

from vecworld.config import STYLE
from vecworld.scenes.prompt import scene_prompt, zoom_crop


def test_scene_prompt():
    p = scene_prompt("ship", ["waves", "gull"])
    assert p.startswith(STYLE) and "a ship in the centre" in p and "waves, gull" in p
    assert "with" not in scene_prompt("ship", [])


def test_zoom_crop_clamps_and_keeps_size():
    img = Image.new("RGB", (512, 512))
    assert zoom_crop(img, 0.98, 0.02).size == (512, 512)  # corner click must not crash


def test_zoom_crop_centres_on_point():
    img = Image.new("RGB", (100, 100), "black")
    img.paste((255, 0, 0), (40, 40, 60, 60))
    out = zoom_crop(img, 0.5, 0.5, frac=0.4)
    assert out.getpixel((50, 50)) == (255, 0, 0)
    assert out.getpixel((2, 2)) == (0, 0, 0)
