# Publishing to the Hugging Face Hub

What goes where:

| Artifact | Size | Home |
|---|---|---|
| Code, configs, results, figures | ~3 MB | GitHub |
| Egyptian test set + synthetic training data | ~3 MB | HF **dataset** repo |
| LoRA adapter (Qwen3-1.7B) | 70 MB | HF **model** repo |
| Encoder (CAMeLBERT, full weights) | 417 MB | HF **model** repo |

Adapters and model weights are gitignored on purpose — a GitHub repo is the wrong place for
half a gigabyte of weights.

## 0. Log in once

```bash
uv run hf auth login        # paste a WRITE token from huggingface.co/settings/tokens
```

## 1. Dataset

```bash
mkdir -p local/hf-dataset && cp \
    data/processed/egy_test.jsonl \
    data/processed/egy_synth_train.jsonl \
    data/processed/egy_synth_dev.jsonl \
    data/annotation/egy_test.csv \
    local/hf-dataset/
cp docs/cards/dataset_card.md local/hf-dataset/README.md

uv run hf repo create egyptian-nlu --type dataset
uv run hf upload Alhasan/egyptian-nlu local/hf-dataset . --repo-type dataset
```

## 2. LoRA adapters

```bash
cp docs/cards/model_card_lora.md adapters/D-qwen3-17b/README.md
uv run hf repo create qwen3-1.7b-egyptian-arabic-lora --type model
uv run hf upload Alhasan/qwen3-1.7b-egyptian-arabic-lora adapters/D-qwen3-17b .
```

The uploaded adapter is the PEFT one (`adapter_model.safetensors` + `adapter_config.json`),
trained with TRL. The 0.6B MLX adapter stays local — it was the on-device experiment, and the
1.7B is the one worth publishing.

## 3. Encoder

```bash
cp docs/cards/model_card_encoder.md models/E-D-camelbert-da/README.md
uv run hf repo create camelbert-egyptian-arabic-nlu --type model
uv run hf upload Alhasan/camelbert-egyptian-arabic-nlu models/E-D-camelbert-da .
```

## 4. Link them together

Once the repos exist, add their URLs to the top of the GitHub README, and put the GitHub URL in
each HF card. Recruiters land on one and follow the link to the other.

## Licensing (get this right)

- **Code** — MIT, see [LICENSE](../LICENSE).
- **Data** — derived from [Amazon MASSIVE](https://github.com/alexa/massive), **CC BY 4.0**. The
  Egyptian rewrites are transformations of MASSIVE utterances, so they inherit CC BY 4.0 and the
  attribution requirement. The dataset card states this; keep it there.
- **Models** — inherit their base model's licence (Qwen3: Apache 2.0; CAMeLBERT: Apache 2.0).
  The LoRA adapters are your own work but are useless without the base weights, so cite them.
