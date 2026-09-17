---
license: apache-2.0
base_model: CAMeL-Lab/bert-base-arabic-camelbert-da
language:
  - ar
tags:
  - arabic
  - egyptian-arabic
  - intent-classification
  - slot-filling
  - token-classification
  - voice-assistant
---

# CAMeLBERT joint intent + slot tagger for Egyptian Arabic

A 110M-parameter encoder with two heads: intent classification on `[CLS]`, and BIO slot tagging
over the tokens. Fine-tuned on [Amazon MASSIVE](https://github.com/alexa/massive) `ar-SA` **plus
6.5k LLM-generated, rule-filtered Egyptian rewrites**.

```
Input : صحيني بكرة الساعة ستة الصبح
Tags  : O      B-date  O       B-time I-time
Output: {"intent": "alarm_set", "slots": {"date": "بكرة", "time": "ستة الصبح"}}
```

## Why this over an LLM

On this task it is the **best model in the comparison**, and by far the cheapest:

| Model | Params | Egyptian test (exact match) | Latency p50 |
|---|---:|---:|---:|
| **This encoder** | 110M | **0.625** | **0.02 s** |
| Qwen3-1.7B + LoRA | 1.7B | 0.590 | — |
| Qwen3-0.6B + LoRA | 596M | 0.510 | 0.55 s |
| Claude Sonnet 5, 5-shot | — | 0.435 | ~1 s (API) |

It cannot invent a slot value — it points at spans in the input — which is exactly why it wins on
a fixed schema. Adding the synthetic Egyptian data was worth **+6.0 points** here (95% CI
[+1.5, +10.5], paired bootstrap), the only place in the project where that gain was significant.

## Usage

The two-head architecture is custom, so load it with the project's code:

```bash
git clone https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants && cd Egyptian-Arabic-understanding-for-voice-assistants && make setup
hf download Alhasan/camelbert-egyptian-arabic-nlu --local-dir models/E-D-camelbert-da
```

```python
from lahja.models.encoder import EncoderPredictor

predict = EncoderPredictor("models/E-D-camelbert-da")
print(predict(["الجو عامل ايه في اسكندرية النهارده"]))
# ['{"intent": "weather_query", "slots": {"place_name": "اسكندرية", "date": "النهارده"}}']
```

Files: `model.pt` (state dict), `meta.json` (base model, intent list, BIO tag list, max length),
plus the tokenizer.

## Training

4 epochs, AdamW lr 5e-5, batch 32, max length 64, linear warmup; best epoch kept by validation
exact match. **Validation was still improving at 4 epochs, so this checkpoint is likely
undertrained** — more epochs should help.

## Limitations

- Egyptian dialect only; other dialects untested.
- Fixed schema: 60 intents, 55 slot types. New intents require retraining, where an LLM would
  only need a prompt change.
- Weakest on `music` / `play` (0.25 / 0.40 exact match), where slots are song and artist names.

Full methodology and figures: **[GitHub repo](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants)**
