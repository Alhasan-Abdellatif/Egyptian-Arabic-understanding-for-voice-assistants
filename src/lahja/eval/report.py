"""Build the results table from results/*/<eval set>/ into markdown.

    python -m lahja.eval.report                      # print + write results/summary.md
    python -m lahja.eval.report --pairs D-qwen3-06b:C-qwen3-06b

"EM clean" excludes test items whose text also appears verbatim in training data (short commands
like "امسح المنبه" that any two writers phrase identically).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lahja import DATA, RESULTS
from lahja.data.normalize import normalize
from lahja.data.schema import load_examples, read_jsonl

EVAL_SETS = ("egy_test", "massive_ar_test")
TRAIN_SOURCES = ("egy_synth_train", "egy_synth_dev", "massive_ar_train")


def training_texts() -> set[str]:
    texts = set()
    for name in TRAIN_SOURCES:
        path = DATA / "processed" / f"{name}.jsonl"
        if path.exists():
            texts |= {normalize(e.utt) for e in load_examples(path)}
    return texts


def overlapping_ids(eval_set: str, train_norm: set[str]) -> set[str]:
    path = DATA / "processed" / f"{eval_set}.jsonl"
    if not path.exists():
        return set()
    return {e.id for e in load_examples(path) if normalize(e.utt) in train_norm}


def collect(eval_set: str, overlap: set[str]) -> list[dict]:
    rows = []
    for run in sorted(RESULTS.iterdir()):
        scores_path = run / eval_set / "scores.json"
        if not scores_path.exists():
            continue
        scores = json.loads(scores_path.read_text())
        items = list(read_jsonl(run / eval_set / "scored.jsonl"))
        clean = [i for i in items if i["id"] not in overlap]
        rows.append(
            {
                "run": run.name,
                "n": scores["n"],
                "json_valid": scores["json_valid"],
                "intent_acc": scores["intent_acc"],
                "slot_f1": scores["slot_f1"],
                "em": scores["exact_match"],
                "ci": scores["exact_match_ci95"],
                "em_clean": sum(i["em"] for i in clean) / max(len(clean), 1),
                "n_clean": len(clean),
                "p50": (scores.get("latency_s") or {}).get("p50"),
            }
        )
    return sorted(rows, key=lambda r: -r["em"])


def paired_delta(run_a: str, run_b: str, eval_set: str, n_boot: int = 2000) -> dict | None:
    """Exact-match difference (a - b) on shared items, with a paired bootstrap CI."""
    try:
        a = {r["id"]: r["em"] for r in read_jsonl(RESULTS / run_a / eval_set / "scored.jsonl")}
        b = {r["id"]: r["em"] for r in read_jsonl(RESULTS / run_b / eval_set / "scored.jsonl")}
    except FileNotFoundError:
        return None
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    da, db = [a[i] for i in ids], [b[i] for i in ids]
    n = len(ids)
    diff = (sum(da) - sum(db)) / n
    rng = random.Random(0)
    boots = sorted(
        (lambda idx: (sum(da[j] for j in idx) - sum(db[j] for j in idx)) / n)(
            [rng.randrange(n) for _ in range(n)]
        )
        for _ in range(n_boot)
    )
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {
        "a": run_a,
        "b": run_b,
        "eval": eval_set,
        "n": n,
        "diff": diff,
        "ci": [lo, hi],
        "significant": lo > 0 or hi < 0,
        "only_a": sum(1 for i in ids if a[i] and not b[i]),
        "only_b": sum(1 for i in ids if b[i] and not a[i]),
    }


def markdown(eval_set: str, rows: list[dict]) -> str:
    out = [
        f"### {eval_set}",
        "",
        "| Model | n | JSON valid | Intent acc | Slot F1 | Exact match | 95% CI | EM clean | p50 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        p50 = f"{r['p50']:.2f}" if r["p50"] else "—"
        out.append(
            f"| {r['run']} | {r['n']} | {r['json_valid']:.2f} | {r['intent_acc']:.3f} | "
            f"{r['slot_f1']:.3f} | **{r['em']:.3f}** | [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | "
            f"{r['em_clean']:.3f} | {p50} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B run names to compare")
    ap.add_argument("--out", default=RESULTS / "summary.md")
    args = ap.parse_args()

    train_norm = training_texts()
    sections = []
    for eval_set in EVAL_SETS:
        overlap = overlapping_ids(eval_set, train_norm)
        rows = collect(eval_set, overlap)
        if rows:
            sections.append(markdown(eval_set, rows))
            sections.append(
                f"\n{len(overlap)} items overlap training data and are excluded from EM clean.\n"
            )

    deltas = [
        d
        for pair in args.pairs
        for eval_set in EVAL_SETS
        if (d := paired_delta(*pair.split(":"), eval_set)) is not None
    ]
    if deltas:
        sections.append("### Paired differences (exact match)\n")
        sections.append(
            "| Comparison | Eval set | n | Δ EM | 95% CI | Significant | Only A | Only B |"
        )
        sections.append("|---|---|---:|---:|---:|---|---:|---:|")
        for d in deltas:
            sections.append(
                f"| {d['a']} − {d['b']} | {d['eval']} | {d['n']} | {d['diff']:+.3f} | "
                f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {'yes' if d['significant'] else 'no'} | "
                f"{d['only_a']} | {d['only_b']} |"
            )

    text = "\n".join(sections) + "\n"
    Path(args.out).write_text(text)
    print(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
