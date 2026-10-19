"""The dreamer: neighbouring places and their in-between scenery from a small local LLM.

Only parsed phrases ever leave this module; raw model output is never logged.
"""
from __future__ import annotations

import re
import threading
from typing import Protocol

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_BULLET = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•]+)\s*")
_PHRASE = re.compile(r"^[^\W_]+(?:[ '\-][^\W_]+)*$")  # letters and digits, words joined by space, ' or -
# Scenery phrases are painted, so they must be visible things.
_NON_VISUAL_STEMS = ("whisper", "laugh", "echo", "sound", "silen", "scent", "smell", "feeling",
                     "memor", "voice", "music")
_NON_VISUAL_WORDS = {"wind", "winds", "windy"}
# Weather and light are painted by the mood, not by a scenery phrase. Rejected
# as the head noun (last word) or as a word of their own, so "a misty pond" and "glowing mushrooms" stay.
_WEATHER_STEMS = ("star", "snowflake", "moonlight", "frost", "glow", "fog", "mist")
_WEATHER_WORDS = {w + s for w in _WEATHER_STEMS for s in ("", "s")}
_CHATTER = re.compile(r"^(?:here are|here is|sure|certainly|okay|ok)\b")
MAX_PHRASE_CHARS = 48
GEN_THREADS = 8  # of 16 logical cores; leave the rest for the step path


def is_visual(phrase: str) -> bool:
    words = phrase.split()
    return not (any(w in _NON_VISUAL_WORDS or w.startswith(_NON_VISUAL_STEMS) for w in words)
                or words[-1].startswith(_WEATHER_STEMS) or not _WEATHER_WORDS.isdisjoint(words))


_EDGE = " \t\"'“”‘’`.,;:!?*_()"


GENERIC_SCENERY = ("a winding path", "a quiet clearing", "a weathered signpost")
MAX_NAME_WORDS = 4
_TEMPLATE = {"name", "phrase", "place"}   # the format line, echoed back as a place


def places_request(place: str, journey: list[str], moods: list[str], n: int) -> list[dict]:
    """The chat messages asking the model for n related neighbouring places, each with 3 in-between sights."""
    mood = f" ({', '.join(moods)})" if moods else ""
    seen = f" Already visited, do not repeat: {', '.join(journey)}." if journey else ""
    return [
        {"role": "system", "content": "You invent places for a painted dreamscape one can wander through."},
        {"role": "user", "content": (
            f'List {n} different places that lie near "{place}"{mood} and relate to it. For each, give '
            "its name (1 to 3 words) and 3 short phrases (2 to 6 words each) naming visible things one "
            "passes on the way there. Only things you could photograph: no sounds, smells, feelings or "
            f"weather.{seen} One place per line in exactly this format, no numbering, no explanations:\n"
            "Name: phrase; phrase; phrase")},
    ]


def _clean(part: str) -> str:
    return " ".join(_BULLET.sub("", part).strip(_EDGE).lower().split())


def parse_places(text: str, n: int) -> list[tuple[str, tuple[str, ...]]]:
    """(name, three scenery phrases) from "Name: phrase; phrase; phrase" lines. Chatter, headers and lines
    without a usable name are dropped; bad scenery phrases are dropped and the rest padded to 3 with
    generic scenery."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line or line.endswith(":") or _CHATTER.match(line.lower()):
            continue
        head, _, tail = line.partition(":")
        name = _clean(head)
        if (not name or not 1 <= len(name.split()) <= MAX_NAME_WORDS or len(name) > MAX_PHRASE_CHARS
                or not _PHRASE.match(name) or name in _TEMPLATE or name in (o[0] for o in out)):
            continue
        phrases: list[str] = []
        for part in tail.split(";"):
            p = _clean(part)
            if (2 <= len(p.split()) <= 8 and len(p) <= MAX_PHRASE_CHARS and _PHRASE.match(p) and is_visual(p)
                    and p not in phrases):
                phrases.append(p)
        phrases = phrases[:3]
        phrases += [g for g in GENERIC_SCENERY if g not in phrases][:3 - len(phrases)]
        out.append((name, tuple(phrases)))
        if len(out) >= n:
            break
    return out


class Dreamer(Protocol):
    def places(self, place: str, journey: list[str], moods: list[str], n: int
               ) -> list[tuple[str, tuple[str, ...]]]: ...


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

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt")
        with self._lock:
            prev = self._torch.get_num_threads()
            self._torch.set_num_threads(GEN_THREADS)
            try:
                with self._torch.no_grad():
                    ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            finally:
                self._torch.set_num_threads(prev)
        return self.tok.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def places(self, place: str, journey: list[str], moods: list[str], n: int
               ) -> list[tuple[str, tuple[str, ...]]]:
        from vecworld.roam.places import fallback_neighbours  # lazy: roam.places must not import this module

        text = self._generate(places_request(place, journey, moods, n), 40 * n + 20)
        seen = {j.lower() for j in journey} | {place.lower()}
        found = [(nm, sc) for nm, sc in parse_places(text, n + len(journey) + 1)
                 if nm not in seen][:n]
        have = {nm for nm, _ in found}
        for nm in fallback_neighbours(place):
            if len(found) >= n:
                break
            if nm not in have and nm not in seen:
                found.append((nm, GENERIC_SCENERY))
        return found
