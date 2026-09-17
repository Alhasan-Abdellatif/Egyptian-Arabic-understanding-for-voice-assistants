"""Inference backends behind one interface: generate(list of chat messages) -> list of texts.

- anthropic: Claude via the API (model A)
- mlx: Apple-silicon inference with mlx-lm, optional LoRA adapter; records per-query latency
- hf: transformers + optional PEFT adapter, batched (Colab GPU)
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


class Backend(Protocol):
    def generate(self, batch: list[list[dict]]) -> list[str]: ...


class AnthropicBackend:
    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "low",
        max_workers: int = 8,
        max_tokens: int = 4000,
    ):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.effort = effort
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        # Token accounting for cost-per-request. Exact when max_workers=1 (as in bench);
        # approximate under concurrency, where Counter updates are not atomic.
        self.usage: Counter = Counter()

    def _one(self, messages: list[dict]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=convo,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            # Server-side refusal fallback; a refused request is re-run on another model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage.update(
            requests=1, input=resp.usage.input_tokens, output=resp.usage.output_tokens
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")

    def generate(self, batch: list[list[dict]]) -> list[str]:
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._one, batch))


def _render(tokenizer, messages: list[dict]) -> str:
    # enable_thinking only affects Qwen3-style templates; other templates ignore it.
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


class MLXBackend:
    def __init__(self, model: str, adapter: str | None = None, max_tokens: int = 128):
        from mlx_lm import load

        self.model, self.tokenizer = load(model, adapter_path=adapter)
        self.max_tokens = max_tokens
        self.latencies: list[float] = []

    def generate(self, batch: list[list[dict]]) -> list[str]:
        from mlx_lm import generate

        outs = []
        for messages in batch:
            prompt = _render(self.tokenizer, messages)
            t0 = time.perf_counter()
            outs.append(
                generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens)
            )
            self.latencies.append(time.perf_counter() - t0)
        return outs


class HFBackend:
    def __init__(
        self, model: str, adapter: str | None = None, max_tokens: int = 128, batch_size: int = 32
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.latencies: list[float] = []

    def generate(self, batch: list[list[dict]]) -> list[str]:
        import torch

        prompts = [_render(self.tokenizer, m) for m in batch]
        outs = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tokenizer(
                prompts[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.model.device)
            chunk = prompts[i : i + self.batch_size]
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False)
            # Amortized per item: a batched call is throughput, not per-request latency.
            # For a real latency number use lahja.eval.bench, which issues one request at a time.
            self.latencies.extend([(time.perf_counter() - t0) / len(chunk)] * len(chunk))
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            outs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outs


class EncoderBackend:
    """Model E: joint intent + BIO encoder. Reads only the user turn; times each query."""

    def __init__(self, model: str, adapter: str | None = None):
        from lahja.models.encoder import EncoderPredictor

        self.predict = EncoderPredictor(model)
        self.latencies: list[float] = []

    def generate(self, batch: list[list[dict]]) -> list[str]:
        outs = []
        for messages in batch:
            t0 = time.perf_counter()
            outs.extend(self.predict([messages[-1]["content"]]))
            self.latencies.append(time.perf_counter() - t0)
        return outs


BACKENDS = {
    "anthropic": AnthropicBackend,
    "mlx": MLXBackend,
    "hf": HFBackend,
    "encoder": EncoderBackend,
}


def make_backend(name: str, **kwargs) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
