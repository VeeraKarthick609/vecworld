"""Discoveries for the dream walk: short phrases from a small local LLM.

Only parsed phrases ever leave this module; raw model output is never logged.
"""
from __future__ import annotations

import random
import re
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
_EDGE = " \t\"'“”‘’`.,;:!?*_()"


def parse_ideas(text: str, limit: int = 8) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            p = _BULLET.sub("", part).strip(_EDGE).lower()
            p = " ".join(p.split())
            if not (2 <= len(p.split()) <= 8) or not _PHRASE.match(p) or p in out:
                continue
            out.append(p)
            if len(out) >= limit:
                return out
    return out


class Dreamer(Protocol):
    def ideas(self, theme: str, moods: list[str], n: int) -> list[str]: ...


class QwenDreamer:
    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu").eval()

    def ideas(self, theme: str, moods: list[str], n: int) -> list[str]:
        mood = ", ".join(moods) if moods else "quiet"
        messages = [
            {"role": "system", "content": "You invent evocative scenery for a painted dreamscape."},
            {"role": "user", "content": (
                f"List {n} short noun phrases (2 to 6 words each) for things one might come upon "
                f'while wandering in "{theme}" ({mood}). One per line, no numbering, no explanations.')},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt")
        with self._torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
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
