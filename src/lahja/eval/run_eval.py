"""Run a model over one or more eval sets; resumable, writes predictions + scores.

    python -m lahja.eval.run_eval --backend anthropic --run-name A_opus5_zeroshot \
        --eval-set data/processed/egy_test.jsonl --eval-set data/processed/massive_ar_test.jsonl --limit 1000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from lahja import RESULTS
from lahja.data.schema import load_examples, read_jsonl, write_jsonl
from lahja.eval.metrics import aggregate, percentiles, score_item
from lahja.models.backends import make_backend
from lahja.models.prompts import (
    SFT_SYSTEM,
    build_messages,
    build_system_prompt,
    load_schema,
    select_shots,
)


def subsample(examples: list, limit: int | None, seed: int = 0) -> list:
    if not limit or limit >= len(examples):
        return examples
    return sorted(random.Random(seed).sample(examples, limit), key=lambda e: e.id)


def weight_files(path: str | None) -> list[list]:
    """Name/size/mtime of a local model or adapter dir's weights (periodic checkpoints excluded)."""
    if not path or not Path(path).is_dir():
        return []
    files = [
        f
        for f in sorted(Path(path).iterdir())
        if f.suffix in {".safetensors", ".pt", ".bin"} and not f.name[0].isdigit()
    ]
    return [[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files]


def fingerprint(args) -> dict:
    """Identifies what produced a prediction, so retrained weights invalidate cached ones."""
    return {
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "prompt": args.prompt,
        "shots": args.shots,
        "weights": weight_files(args.adapter) + weight_files(args.model),
    }


def score_and_write(golds, done: dict[str, str | None], out_dir: Path, backend=None) -> dict:
    """Score raw predictions and write scored.jsonl + scores.json. Shared with the batch runner."""
    items = [{"id": g.id, **score_item(g, done.get(g.id))} for g in golds]
    write_jsonl(out_dir / "scored.jsonl", items)
    scores = aggregate(items)
    latencies = sorted(getattr(backend, "latencies", None) or [])
    if latencies:  # only local backends time themselves; resumed items are not re-timed
        scores["latency_s"] = percentiles(latencies)
        backend.latencies.clear()
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def evaluate_set(
    backend,
    golds,
    system,
    shots,
    out_dir: Path,
    chunk: int,
    fp: dict | None = None,
    fresh: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    fp_path = out_dir / "fingerprint.json"
    if preds_path.exists() and not fresh:
        cached = json.loads(fp_path.read_text()) if fp_path.exists() else None
        if fp is not None and cached != fp:
            print(f"{out_dir.name}: model or prompt changed since last run - re-predicting")
            fresh = True
    if fresh and preds_path.exists():
        preds_path.unlink()
    if fp is not None:
        fp_path.write_text(json.dumps(fp, indent=2) + "\n")

    done = {r["id"]: r["raw"] for r in read_jsonl(preds_path)} if preds_path.exists() else {}
    todo = [g for g in golds if g.id not in done]
    for i in tqdm(range(0, len(todo), chunk), desc=out_dir.name, disable=not todo):
        part = todo[i : i + chunk]
        raws = backend.generate([build_messages(g.utt, system, shots) for g in part])
        write_jsonl(
            preds_path,
            ({"id": g.id, "raw": r} for g, r in zip(part, raws, strict=True)),
            append=True,
        )
        done.update({g.id: r for g, r in zip(part, raws, strict=True)})

    return score_and_write(golds, done, out_dir, backend)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--eval-set", action="append", required=True)
    ap.add_argument("--limit", type=int, default=None, help="random subsample per eval set")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--shots-from", default="data/processed/massive_ar_train.jsonl")
    ap.add_argument(
        "--prompt",
        choices=["full", "sft"],
        default="full",
        help="sft = short prompt used in training",
    )
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (mlx / hf)")
    ap.add_argument("--effort", default="low", help="anthropic only")
    ap.add_argument("--workers", type=int, default=8, help="anthropic only")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument(
        "--fresh", action="store_true", help="discard cached predictions even if they still match"
    )
    args = ap.parse_args()

    if args.backend == "anthropic":
        kwargs = {"max_workers": args.workers, "effort": args.effort}
    else:
        if not args.model:
            ap.error(f"--model is required for the {args.backend} backend")
        kwargs = {"adapter": args.adapter}
    if args.model:
        kwargs["model"] = args.model
    backend = make_backend(args.backend, **kwargs)
    system = SFT_SYSTEM if args.prompt == "sft" else build_system_prompt(load_schema())
    shots = select_shots(load_examples(args.shots_from), args.shots) if args.shots else []

    run_dir = RESULTS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    fp = fingerprint(args)
    for path in args.eval_set:
        golds = subsample(load_examples(path), args.limit)
        scores = evaluate_set(
            backend, golds, system, shots, run_dir / Path(path).stem, args.chunk, fp, args.fresh
        )
        print(f"{Path(path).stem}: " + json.dumps(scores))


if __name__ == "__main__":
    main()
