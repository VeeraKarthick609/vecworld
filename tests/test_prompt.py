from PIL import Image

from vecworld.config import STYLE
from vecworld.scenes.prompt import scene_prompt, zoom_crop


def test_concrete_prompt_is_a_plain_scene():
    p = scene_prompt("ship", ["waves", "gull"], concrete=True)
    assert p.startswith("a scene of a ship in the centre, with waves, gull around it")
    assert p.endswith(STYLE)
    assert "with" not in scene_prompt("ship", [], concrete=True).replace(STYLE, "")


def test_vowel_gets_an():
    assert scene_prompt("elephant", [], concrete=True).startswith("a scene of an elephant")


def test_abstract_prompt_is_symbolic():
    p = scene_prompt("anger", ["rage", "grief"], concrete=False)
    assert p.startswith("a symbolic scene evoking anger, with rage, grief") and p.endswith(STYLE)


def test_no_landscape_framing():
    assert "landscape" not in STYLE
    assert "landscape" not in scene_prompt("anger", ["rage"], concrete=False)


def test_zoom_crop_clamps_and_keeps_size():
    img = Image.new("RGB", (512, 512))
    assert zoom_crop(img, 0.98, 0.02).size == (512, 512)  # corner click must not crash


def test_zoom_crop_centres_on_point():
    img = Image.new("RGB", (100, 100), "black")
    img.paste((255, 0, 0), (40, 40, 60, 60))
    out = zoom_crop(img, 0.5, 0.5, frac=0.4)
    assert out.getpixel((50, 50)) == (255, 0, 0)
    assert out.getpixel((2, 2)) == (0, 0, 0)
