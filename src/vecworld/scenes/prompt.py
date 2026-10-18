from PIL import Image

from vecworld.config import STYLE


def scene_prompt(centre: str, others: list[str], concrete: bool = True) -> str:
    """Subject first so CLIP weights it; abstract concepts get a symbolic scene instead of a misty void."""
    with_ = f", with {', '.join(others)}" if others else ""
    if concrete:
        article = "an" if centre[0] in "aeiou" else "a"
        subject = f"a scene of {article} {centre} in the centre{with_ + ' around it' if others else ''}"
    else:
        subject = f"a symbolic scene evoking {centre}{with_}"
    return f"{subject}, {STYLE}"


def zoom_crop(img: Image.Image, cx: float, cy: float, frac: float = 0.6) -> Image.Image:
    """Square crop of side frac * width centred on (cx, cy), kept inside the image, scaled back up."""
    w, h = img.size
    side = int(min(w, h) * frac)
    left = min(max(int(cx * w - side / 2), 0), w - side)
    top = min(max(int(cy * h - side / 2), 0), h - side)
    return img.crop((left, top, left + side, top + side)).resize((w, h), Image.LANCZOS)
