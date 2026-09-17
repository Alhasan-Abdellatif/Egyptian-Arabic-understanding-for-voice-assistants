"""Train the joint intent + BIO encoder baseline (model E) on a data mix; keeps the best epoch.

    python -m lahja.train.train_encoder --model CAMeL-Lab/bert-base-arabic-camelbert-da \
        --mix D --out models/E-D-camelbert-da
"""

from __future__ import annotations

import argparse
import random
import time

from lahja import DATA
from lahja.data.bio import bio_labels, tag_set
from lahja.data.schema import Example, load_examples
from lahja.eval.metrics import aggregate, score_item
from lahja.models.prompts import load_schema

MIXES = {
    "C": (["massive_ar_train"], ["massive_ar_dev"]),
    "D": (["massive_ar_train", "egy_synth_train"], ["massive_ar_dev", "egy_synth_dev"]),
}


def _load(names: list[str]) -> list[Example]:
    return [e for n in names for e in load_examples(DATA / "processed" / f"{n}.jsonl")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mix", required=True, choices=list(MIXES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--valid-per-source", type=int, default=250)
    ap.add_argument("--limit-train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from lahja.models.encoder import JointEncoder, pick_device, predict_json, save_encoder

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    schema = load_schema()
    intents, tags = schema["intents"], tag_set(schema["slot_types"])
    intent_id = {x: i for i, x in enumerate(intents)}
    tag_id = {x: i for i, x in enumerate(tags)}

    train_sources, valid_sources = MIXES[args.mix]
    train = _load(train_sources)[: args.limit_train]
    valid = []
    for name in valid_sources:
        exs = _load([name])
        valid += rng.sample(exs, min(args.valid_per_source, len(exs)))

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointEncoder(args.model, len(intents), len(tags)).to(device)

    def batchify(exs: list[Example]):
        enc = tokenizer(
            [e.utt for e in exs],
            truncation=True,
            max_length=args.max_len,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"].tolist()
        labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for i, ex in enumerate(exs):
            for j, tag in enumerate(bio_labels(ex, offsets[i])):
                if tag is not None:
                    labels[i, j] = tag_id[tag]
        y_intent = torch.tensor([intent_id[e.intent] for e in exs])
        return (t.to(device) for t in (enc["input_ids"], enc["attention_mask"], y_intent, labels))

    n_batches = (len(train) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * n_batches
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    meta = {"base_model": args.model, "intents": intents, "tags": tags, "max_len": args.max_len}
    best = -1.0
    print(f"device={device} train={len(train)} valid={len(valid)} steps={total_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        t0, running = time.time(), 0.0
        for b in range(0, len(train), args.batch_size):
            ids, mask, y_intent, y_tags = batchify(train[b : b + args.batch_size])
            intent_logits, slot_logits = model(ids, mask)
            loss = ce(intent_logits, y_intent) + ce(
                slot_logits.reshape(-1, slot_logits.size(-1)), y_tags.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()

        preds = predict_json(
            model, tokenizer, [e.utt for e in valid], intents, tags, args.max_len, device
        )
        scores = aggregate([score_item(g, p) for g, p in zip(valid, preds, strict=True)])
        print(
            f"epoch {epoch}: loss={running / n_batches:.4f} valid EM={scores['exact_match']:.3f} "
            f"intent={scores['intent_acc']:.3f} slotF1={scores['slot_f1']:.3f} ({time.time() - t0:.0f}s)"
        )
        if scores["exact_match"] > best:
            best = scores["exact_match"]
            save_encoder(model, tokenizer, args.out, {**meta, "epoch": epoch, "valid": scores})
    print(f"best valid EM={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
