"""Run the Egyptian-Arabic NLU model on a sentence.

    python -m lahja.predict "صحيني بكرة الساعة ستة الصبح"
    python -m lahja.predict --model lora "الجو عامل ايه في اسكندرية النهارده"
    echo "شغل اغاني لعمرو دياب" | python -m lahja.predict

Weights are downloaded from the Hugging Face Hub on first use and cached. Pass a local
directory to `--weights` to use your own trained checkpoint instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENCODER_REPO = "Alhasan/camelbert-egyptian-arabic-nlu"
LORA_BASE = "Qwen/Qwen3-1.7B"
LORA_REPO = "Alhasan/qwen3-1.7b-egyptian-arabic-lora"


def resolve(weights: str) -> str:
    """A local path if it exists, otherwise download the Hub repo and return its cache path."""
    if Path(weights).exists():
        return weights
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=weights)


class Predictor:
    """Loads once, predicts many. `model` is 'encoder' (110M, fast) or 'lora' (Qwen3-1.7B)."""

    def __init__(self, model: str = "encoder", weights: str | None = None):
        self.model = model
        if model == "encoder":
            from lahja.models.encoder import EncoderPredictor

            self._predict = EncoderPredictor(resolve(weights or ENCODER_REPO))
        elif model == "lora":
            from lahja.models.backends import make_backend

            adapter = resolve(weights or LORA_REPO)
            try:  # Apple silicon: mlx is much faster; fall back to torch elsewhere
                import mlx.core  # noqa: F401

                backend = "mlx"
            except ImportError:
                backend = "hf"
            self._backend = make_backend(backend, model=LORA_BASE, adapter=adapter)
        else:
            raise ValueError(f"model must be 'encoder' or 'lora', got {model!r}")

    def __call__(self, texts: list[str]) -> list[dict]:
        from lahja.eval.metrics import parse_prediction

        if self.model == "encoder":
            raw = self._predict(texts)
        else:
            from lahja.models.prompts import SFT_SYSTEM, build_messages

            raw = self._backend.generate([build_messages(t, SFT_SYSTEM) for t in texts])
        out = []
        for text, r in zip(texts, raw, strict=True):
            parsed = parse_prediction(r)
            out.append(
                {
                    "text": text,
                    "intent": parsed["intent"] if parsed else None,
                    "slots": dict(parsed["pairs"]) if parsed else {},
                }
            )
        return out


def predict(
    texts: list[str] | str, model: str = "encoder", weights: str | None = None
) -> list[dict]:
    """One-shot helper. For many calls, build a Predictor once and reuse it."""
    return Predictor(model, weights)([texts] if isinstance(texts, str) else texts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("text", nargs="*", help="one or more commands; omit to read stdin")
    ap.add_argument("--model", choices=["encoder", "lora"], default="encoder")
    ap.add_argument("--weights", default=None, help="local checkpoint dir or Hub repo id")
    ap.add_argument("--compact", action="store_true", help="one JSON object per line")
    args = ap.parse_args()

    texts = args.text or [ln.strip() for ln in sys.stdin if ln.strip()]
    if not texts:
        ap.error("give a command as an argument or on stdin")

    for row in Predictor(args.model, args.weights)(texts):
        print(json.dumps(row, ensure_ascii=False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
