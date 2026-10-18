from typing import Protocol

import numpy as np

from vecworld.config import LAYER, MAX_TEXT_CHARS

# Asking the model to compress the text into one word makes the last token's
# hidden state carry the meaning; plain mean-pooling of a bare word is mostly token noise.
TEMPLATE = 'This word: "{}" means in one word:"'


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LLMEmbedder:
    """Last-token hidden state of one layer of a small causal LLM, prompted to summarise the text."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B", layer: int = LAYER, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.layer = layer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        # fp32 everywhere, so vectors embedded on the GPU match ones embedded on the CPU.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        prompts = [TEMPLATE.format(t[:MAX_TEXT_CHARS]) for t in texts]
        out = []
        with torch.inference_mode():
            for i in range(0, len(prompts), 64):
                batch = self.tok(prompts[i:i + 64], return_tensors="pt", padding=True).to(self.device)
                hidden = self.model(**batch, output_hidden_states=True).hidden_states[self.layer]
                out.append(hidden[:, -1].float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
