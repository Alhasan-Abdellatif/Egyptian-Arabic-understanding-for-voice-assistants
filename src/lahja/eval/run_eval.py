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

from lahja import DATA, RESULTS
from lahja.data.schema import load_examples, read_jsonl, write_jsonl
from lahja.eval.metrics import aggregate, score_item
from lahja.models.backends import make_backend
from lahja.models.prompts import build_messages, build_system_prompt, load_schema, select_shots


def subsample(examples: list, limit: int | None, seed: int = 0) -> list:
    if not limit or limit >= len(examples):
        return examples
    return sorted(random.Random(seed).sample(examples, limit), key=lambda e: e.id)


def evaluate_set(backend, golds, system, shots, out_dir: Path, chunk: int) -> dict:
    preds_path = out_dir / "predictions.jsonl"
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

    items = [{"id": g.id, **score_item(g, done[g.id])} for g in golds]
    write_jsonl(out_dir / "scored.jsonl", items)
    scores = aggregate(items)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--eval-set", action="append", required=True)
    ap.add_argument("--limit", type=int, default=None, help="random subsample per eval set")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--shots-from", default=DATA / "processed/massive_ar_train.jsonl")
    ap.add_argument("--effort", default="low", help="anthropic only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    kwargs = {"max_workers": args.workers, "effort": args.effort}
    if args.model:
        kwargs["model"] = args.model
    backend = make_backend(args.backend, **kwargs)
    system = build_system_prompt(load_schema())
    shots = select_shots(load_examples(args.shots_from), args.shots) if args.shots else []

    run_dir = RESULTS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    for path in args.eval_set:
        golds = subsample(load_examples(path), args.limit)
        scores = evaluate_set(backend, golds, system, shots, run_dir / Path(path).stem, args.chunk)
        print(f"{Path(path).stem}: " + json.dumps(scores))


if __name__ == "__main__":
    main()
