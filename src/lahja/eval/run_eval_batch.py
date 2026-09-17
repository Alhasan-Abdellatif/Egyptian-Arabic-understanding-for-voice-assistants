"""Evaluate a Claude model through the Message Batches API (50% cheaper; results within ~1h).

    python -m lahja.eval.run_eval_batch submit --run-name A_sonnet5_zeroshot \
        --eval-set data/processed/egy_test.jsonl --eval-set data/processed/massive_ar_test.jsonl --limit 300
    python -m lahja.eval.run_eval_batch status  --run-name A_sonnet5_zeroshot
    python -m lahja.eval.run_eval_batch collect --run-name A_sonnet5_zeroshot

Writes the same predictions.jsonl / scored.jsonl / scores.json layout as run_eval, so both
paths are scored by identical code. Server-side refusal fallbacks are not available on the
Batches API; a refused request is recorded as an empty prediction and scored as wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lahja import RESULTS
from lahja.data.schema import load_examples, read_jsonl, write_jsonl
from lahja.eval.run_eval import score_and_write, subsample
from lahja.models.prompts import (
    SFT_SYSTEM,
    build_messages,
    build_system_prompt,
    load_schema,
    select_shots,
)

SEPARATOR = "__"  # custom_id = "<eval set stem>__<example id>"


def _eval_sets(args) -> dict[str, list]:
    return {Path(p).stem: subsample(load_examples(p), args.limit) for p in args.eval_set}


def _state_path(run_name: str) -> Path:
    return RESULTS / run_name / "batch.json"


def cmd_submit(args, client) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    system = SFT_SYSTEM if args.prompt == "sft" else build_system_prompt(load_schema())
    shots = select_shots(load_examples(args.shots_from), args.shots) if args.shots else []
    requests = []
    for stem, golds in _eval_sets(args).items():
        for ex in golds:
            messages = build_messages(ex.utt, system, shots)
            requests.append(
                Request(
                    custom_id=f"{stem}{SEPARATOR}{ex.id}",
                    params=MessageCreateParamsNonStreaming(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        system=[m["content"] for m in messages if m["role"] == "system"][0],
                        messages=[m for m in messages if m["role"] != "system"],
                        output_config={"effort": args.effort},
                    ),
                )
            )
    batch = client.messages.batches.create(requests=requests)
    run_dir = RESULTS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _state_path(args.run_name).write_text(
        json.dumps({"batch_id": batch.id, "n_requests": len(requests), **vars(args)}, default=str)
        + "\n"
    )
    print(f"submitted batch {batch.id} with {len(requests)} requests")


def cmd_status(args, client) -> None:
    state = json.loads(_state_path(args.run_name).read_text())
    batch = client.messages.batches.retrieve(state["batch_id"])
    print(batch.processing_status, batch.request_counts)


def _records_from_file(path: str):
    """(custom_id, outcome, stop_reason, text) from a results .jsonl downloaded from the console."""
    for row in read_jsonl(path):
        result = row["result"]
        message = result.get("message") or {}
        blocks = message.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        yield row["custom_id"], result["type"], message.get("stop_reason"), text


def _records_from_api(client, batch_id: str):
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        sys.exit(f"batch still {batch.processing_status}: {batch.request_counts}")
    for result in client.messages.batches.results(batch_id):
        text, stop = "", None
        if result.result.type == "succeeded":
            message = result.result.message
            stop = message.stop_reason
            text = "".join(b.text for b in message.content if b.type == "text")
        yield result.custom_id, result.result.type, stop, text


def cmd_collect(args, client) -> None:
    state = json.loads(_state_path(args.run_name).read_text())
    records = (
        _records_from_file(args.results_file)
        if args.results_file
        else _records_from_api(client, state["batch_id"])
    )

    raws: dict[str, dict[str, str]] = {}
    outcomes: Counter = Counter()
    for custom_id, outcome, stop_reason, text in records:
        outcomes[outcome] += 1
        if stop_reason:
            outcomes[f"stop_{stop_reason}"] += 1
        stem, _, example_id = custom_id.partition(SEPARATOR)
        raws.setdefault(stem, {})[example_id] = "" if stop_reason == "refusal" else text
    print("outcomes:", dict(outcomes))

    args.eval_set = state["eval_set"]
    args.limit = state["limit"]
    for stem, golds in _eval_sets(args).items():
        out_dir = RESULTS / args.run_name / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        done = raws.get(stem, {})
        write_jsonl(
            out_dir / "predictions.jsonl", ({"id": g.id, "raw": done.get(g.id)} for g in golds)
        )
        scores = score_and_write(golds, done, out_dir)
        print(f"{stem}: " + json.dumps(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--run-name", required=True)
    submit.add_argument("--eval-set", action="append", required=True)
    submit.add_argument("--model", default="claude-sonnet-5")
    submit.add_argument("--limit", type=int, default=None)
    submit.add_argument("--shots", type=int, default=0)
    submit.add_argument("--shots-from", default="data/processed/massive_ar_train.jsonl")
    submit.add_argument("--prompt", choices=["full", "sft"], default="full")
    submit.add_argument("--effort", default="low")
    submit.add_argument("--max-tokens", type=int, default=4000)
    for name in ("status", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--run-name", required=True)
        if name == "collect":
            p.add_argument(
                "--results-file", help="results .jsonl downloaded from the console (skips the API)"
            )
    args = ap.parse_args()

    client = None
    if not getattr(args, "results_file", None):
        import anthropic

        client = anthropic.Anthropic(max_retries=5)
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](args, client)


if __name__ == "__main__":
    main()
