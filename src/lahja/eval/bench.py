"""Measure per-request latency (and API cost) for any backend, one request at a time.

    python -m lahja.eval.bench --backend encoder --model models/E-D-camelbert-da --name E-D
    python -m lahja.eval.bench --backend mlx --model Qwen/Qwen3-0.6B \
        --adapter adapters/D-qwen3-06b --prompt sft --name D-0.6B
    python -m lahja.eval.bench --backend anthropic --model claude-sonnet-5 --n 30 --name A-sonnet5

Latency numbers only mean something when every model is measured **on the same machine**, so
run these back to back on one host and report them together. Requests are issued sequentially
(concurrency 1): this measures per-request latency, not throughput.

API latency is wall-clock time from request to full response, so it includes network round trip
and depends on your connection and on server load - label it as measured-from-here, not as a
property of the model. The Batches API has no meaningful latency (it is asynchronous by design);
benchmark the frontier model through the normal synchronous endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time

from lahja import RESULTS
from lahja.data.schema import load_examples
from lahja.eval.metrics import percentiles
from lahja.models.backends import make_backend
from lahja.models.prompts import SFT_SYSTEM, build_messages, build_system_prompt, load_schema

# USD per million tokens, synchronous pricing (Batches API is 50% of these).
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", choices=["full", "sft"], default="full")
    ap.add_argument("--name", required=True, help="label for the benchmark row")
    ap.add_argument("--n", type=int, default=50, help="requests to time")
    ap.add_argument("--warmup", type=int, default=3, help="untimed requests first")
    ap.add_argument("--eval-set", default="data/processed/egy_test.jsonl")
    args = ap.parse_args()

    examples = load_examples(args.eval_set)
    rng = random.Random(0)
    sample = rng.sample(examples, min(args.n + args.warmup, len(examples)))
    system = SFT_SYSTEM if args.prompt == "sft" else build_system_prompt(load_schema())

    kwargs = {"model": args.model}
    if args.backend == "anthropic":
        kwargs["max_workers"] = 1  # sequential: we are measuring latency, not throughput
    else:
        kwargs["adapter"] = args.adapter
    backend = make_backend(args.backend, **kwargs)

    latencies = []
    for i, ex in enumerate(sample):
        messages = build_messages(ex.utt, system)
        t0 = time.perf_counter()
        backend.generate([messages])
        elapsed = time.perf_counter() - t0
        if i >= args.warmup:  # discard warmup (weight loading, first-call compilation)
            latencies.append(elapsed)

    row = {
        "name": args.name,
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "prompt": args.prompt,
        "n": len(latencies),
        "warmup": args.warmup,
        "latency_s": percentiles(latencies),
    }

    usage = getattr(backend, "usage", None)
    if usage and usage.get("requests"):
        price_in, price_out = PRICES.get(args.model, (0.0, 0.0))
        per_req = (
            (usage["input"] * price_in + usage["output"] * price_out) / 1e6 / usage["requests"]
        )
        row["tokens_per_request"] = {
            "input": round(usage["input"] / usage["requests"], 1),
            "output": round(usage["output"] / usage["requests"], 1),
        }
        row["usd_per_1k_requests"] = round(per_req * 1000, 2)
        row["usd_per_1k_requests_batch"] = round(per_req * 1000 / 2, 2)

    out = RESULTS / "benchmarks" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
