"""Joint intent + slot encoder (BERT-style), used as model E. Requires torch + transformers.

Predictions are rendered to the same JSON the LLMs produce, so every model is scored by the
same metrics code.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from lahja.data.bio import decode_spans
from lahja.data.formats import target_json


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class JointEncoder(nn.Module):
    def __init__(self, base_model: str, n_intents: int, n_tags: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.slot_head = nn.Linear(hidden, n_tags)

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        states = self.dropout(states)
        return self.intent_head(states[:, 0]), self.slot_head(states)


@torch.no_grad()
def predict_json(
    model: JointEncoder,
    tokenizer,
    utts: list[str],
    intents: list[str],
    tags: list[str],
    max_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> list[str]:
    model.eval()
    outs = []
    for b in range(0, len(utts), batch_size):
        chunk = utts[b : b + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"].tolist()
        intent_logits, slot_logits = model(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        intent_ids = intent_logits.argmax(-1).tolist()
        tag_ids = slot_logits.argmax(-1).tolist()
        for i, utt in enumerate(chunk):
            spans = decode_spans(utt, offsets[i], [tags[t] for t in tag_ids[i]])
            outs.append(target_json(intents[intent_ids[i]], spans))
    return outs


def save_encoder(model: JointEncoder, tokenizer, out_dir: str | Path, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tokenizer.save_pretrained(out)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


class EncoderPredictor:
    def __init__(self, model_dir: str | Path, device: torch.device | None = None):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        self.intents, self.tags, self.max_len = meta["intents"], meta["tags"], meta["max_len"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = JointEncoder(meta["base_model"], len(self.intents), len(self.tags))
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.device = device or pick_device()
        self.model.to(self.device).eval()

    def __call__(self, utts: list[str]) -> list[str]:
        return predict_json(
            self.model, self.tokenizer, utts, self.intents, self.tags, self.max_len, self.device
        )
