from PIL import Image

from vecworld.config import STYLE


def scene_prompt(centre: str, others: list[str]) -> str:
    prompt = f"{STYLE}, a {centre} in the centre"
    if others:
        prompt += f", with {', '.join(others)} around it"
    return prompt


def zoom_crop(img: Image.Image, cx: float, cy: float, frac: float = 0.6) -> Image.Image:
    """Square crop of side frac * width centred on (cx, cy), kept inside the image, scaled back up."""
    w, h = img.size
    side = int(min(w, h) * frac)
    left = min(max(int(cx * w - side / 2), 0), w - side)
    top = min(max(int(cy * h - side / 2), 0), h - side)
    return img.crop((left, top, left + side, top + side)).resize((w, h), Image.LANCZOS)
