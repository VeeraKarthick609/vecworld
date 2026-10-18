"""Discoveries for the dream walk: short phrases from a small local LLM.

Only parsed phrases ever leave this module; raw model output is never logged.
"""
from __future__ import annotations

import random
import re
import threading
from typing import Protocol

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

FALLBACK_IDEAS: tuple[str, ...] = (
    "an old lantern on a post",
    "a ruined stone archway",
    "a still pond",
    "a lone wooden bench",
    "a crooked wooden fence",
    "a mossy stone well",
    "a narrow footbridge",
    "a weathered signpost",
    "a small stone shrine",
    "an abandoned cart",
    "a twisted old tree",
    "a flickering campfire",
)

_BULLET = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•]+)\s*")
_PHRASE = re.compile(r"^[^\W_]+(?:[ '\-][^\W_]+)*$")  # letters+digits: the brief test needs "thing number 7"
# Discoveries are painted and then located by CLIPSeg, so they must be visible things.
_NON_VISUAL_STEMS = ("whisper", "laugh", "echo", "sound", "silen", "scent", "smell", "feeling",
                     "memor", "voice", "music")
_NON_VISUAL_WORDS = {"wind", "winds", "windy"}
# Weather and light are painted by the mood, and CLIPSeg can't box them: not discoveries.
# Matched on the head noun (last word), so "a misty pond" and "glowing mushrooms" stay.
_WEATHER_STEMS = ("star", "snowflake", "moonlight", "frost", "glow", "fog", "mist")
_CHATTER = re.compile(r"^(?:here are|here is|sure|certainly|okay|ok)\b")
MAX_PHRASE_CHARS = 48
GEN_THREADS = 8  # of 16 logical cores; leave the rest for CLIPSeg on the step path


def is_visual(phrase: str) -> bool:
    words = phrase.split()
    return not (any(w in _NON_VISUAL_WORDS or w.startswith(_NON_VISUAL_STEMS) for w in words)
                or words[-1].startswith(_WEATHER_STEMS))


_EDGE = " \t\"'“”‘’`.,;:!?*_()"


def parse_ideas(text: str, limit: int = 8) -> list[str]:
    """Short visible-object phrases from the model's list. Chatter lines ("Here are...", "Objects:") are
    dropped; a one- or two-line answer is a comma list, otherwise each line's text after a comma is a
    description of it ("a stone well, covered in moss") and is dropped too."""
    out: list[str] = []
    lines = [l for l in (_BULLET.sub("", l).strip() for l in text.splitlines()) if l]
    lines = [l for l in lines if not (l.endswith(":") or _CHATTER.match(l.lower()))]
    split = len([l for l in text.splitlines() if l.strip()]) < 3
    for line in lines:
        for part in line.split(",") if split else line.split(",")[:1]:
            p = _BULLET.sub("", part).strip(_EDGE).lower()
            p = " ".join(p.split())
            if (not (2 <= len(p.split()) <= 8) or len(p) > MAX_PHRASE_CHARS or not _PHRASE.match(p) or p in out
                    or not is_visual(p)):
                continue
            out.append(p)
            if len(out) >= limit:
                return out
    return out


def ideas_request(theme: str, moods: list[str], n: int) -> list[dict]:
    """The chat messages asking the model for n discoveries."""
    mood = f" ({', '.join(moods)})" if moods else ""
    return [
        {"role": "system", "content": "You invent evocative scenery for a painted dreamscape."},
        {"role": "user", "content": (
            f"List {n} short noun phrases (2 to 6 words each) naming concrete, visible objects or "
            f'landmarks one might come upon while wandering in "{theme}"{mood}. Only things you '
            "could photograph: no sounds, smells, feelings or weather. "
            "One per line, no numbering, no explanations.")},
    ]


class Dreamer(Protocol):
    def ideas(self, theme: str, moods: list[str], n: int) -> list[str]: ...


class QwenDreamer:
    """Hold ONE instance per process (fp32 1.5B is ~6 GB of RAM). Calls are serialized by a lock
    and generation is capped to GEN_THREADS torch threads."""

    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._lock = threading.Lock()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to("cpu").eval()

    def ideas(self, theme: str, moods: list[str], n: int) -> list[str]:
        messages = ideas_request(theme, moods, n)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt")
        with self._lock:
            prev = self._torch.get_num_threads()
            self._torch.set_num_threads(GEN_THREADS)
            try:
                with self._torch.no_grad():
                    ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
            finally:
                self._torch.set_num_threads(prev)
        text = self.tok.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        found = parse_ideas(text, n)
        if len(found) < n:
            pool = list(FALLBACK_IDEAS)
            random.shuffle(pool)
            for p in pool:
                if len(found) >= n:
                    break
                if p not in found:
                    found.append(p)
        return found
