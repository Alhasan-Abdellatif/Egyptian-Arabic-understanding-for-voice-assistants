# Egyptian Arabic understanding for voice assistants

[![ci](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants/actions/workflows/ci.yml/badge.svg)](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants/actions/workflows/ci.yml)

Turn an Egyptian Arabic voice command into a structured tool call — one intent plus its slots —
using **LoRA fine-tuned Qwen3**, a **synthetic dialect corpus generated with Claude**, and a
**native-written benchmark**. Everything is measured against a frontier LLM and a small encoder
baseline, with paired significance tests.

**Models & data on Hugging Face:**
[dataset](https://huggingface.co/datasets/Alhasan/egyptian-nlu) ·
[Qwen3-1.7B LoRA](https://huggingface.co/Alhasan/qwen3-1.7b-egyptian-arabic-lora) ·
[CAMeLBERT encoder](https://huggingface.co/Alhasan/camelbert-egyptian-arabic-nlu)

```
صحيني بكرة الساعة ستة الصبح
    → {"intent": "alarm_set", "slots": {"date": "بكرة", "time": "ستة الصبح"}}
```

## What it does

Real predictions from the fine-tuned Qwen3-1.7B — [more in docs/examples.md](docs/examples.md):

| Spoken Egyptian input | Model output |
|---|---|
| شغل ببجي | `{"intent": "play_game", "slots": {"game_name": "ببجي"}}` |
| اتصل على أقرب مطعم مندي عندهم توصيل | `{"intent": "takeaway_order", "slots": {"business_type": "مطعم", "food_type": "مندي", "order_type": "توصيل"}}` |
| ايه عندي في قائمة المهام بتاعتى | `{"intent": "lists_query", "slots": {"list_name": "المهام"}}` |

60 intents, 55 slot types, **100% valid JSON** with plain greedy decoding — no constrained decoding.

## Results

**[Full report with figures → docs/results.md](docs/results.md)**

![Exact match on the Egyptian test set](results/figures/headline_egy.png)

- **Fine-tuning beats prompting on this task.** Qwen3-1.7B + LoRA scores **0.590** exact match
  against **0.435** for Claude Sonnet 5 with five examples — **+15.5 points**, 95% CI
  [+8.5, +22.5], paired bootstrap. Showing the frontier model five demonstrations does *not*
  close the gap: its errors are the dataset's labelling conventions, not Arabic comprehension.
- **Synthetic dialect data helps the model that lacks dialect knowledge.** A 110M dialect-BERT
  gains a significant **+6.0 points** from it; Qwen3-0.6B/1.7B gain nothing measurable, because
  they already carry Egyptian from pretraining. On dialect-marked sentences the LLMs do gain
  ~+6 too — the aggregate hides it.
- **Small beats large here.** That 110M encoder (0.625) matches the 1.7B LLM (0.590, n.s.) at
  **20 ms/query on-device** versus ~1 s for the API.

## Models and data

Weights and datasets live on the Hub; this repo holds the code, configs and results.

| Artifact | What it is | Size |
|---|---|---:|
| [`Alhasan/egyptian-nlu`](https://huggingface.co/datasets/Alhasan/egyptian-nlu) | 200 hand-written Egyptian test commands + 7.2k filtered synthetic training rows | 3 MB |
| [`Alhasan/qwen3-1.7b-egyptian-arabic-lora`](https://huggingface.co/Alhasan/qwen3-1.7b-egyptian-arabic-lora) | Best accuracy — 0.590 exact match | 70 MB |
| [`Alhasan/camelbert-egyptian-arabic-nlu`](https://huggingface.co/Alhasan/camelbert-egyptian-arabic-nlu) | 110M joint intent + BIO tagger — best overall (0.625) at 20 ms/query | 417 MB |

Each carries a model or dataset card with the exact prompt format, results and limitations.
Publishing steps: [docs/huggingface.md](docs/huggingface.md).

## Quickstart

```bash
make setup && make download && make data     # MASSIVE ar-SA -> data/processed
make pilot && make submit && make collect    # generate Egyptian rewrites (Claude, Batches API)
make synth && make sft                       # filter -> data/sft/{C,D}
make train-mlx-D && make train-enc-D         # train (Apple silicon; see docs/ for GPU)
make eval-mlx-D && make eval-enc-D
make report && make figures && make examples
```

## How it works

| Stage | What happens |
|---|---|
| **Data** | [Amazon MASSIVE](https://github.com/alexa/massive) `ar-SA` (Saudi/MSA) parsed into intent + slot spans |
| **Generation** | Claude Sonnet 5 rewrites 4k seeds into Egyptian (2 variants each) in MASSIVE bracket notation |
| **Filtering** | Rule-based only — slot preservation, copy detection, near-duplicate removal. 8,002 → **7,239** kept |
| **Benchmark** | 200 Egyptian commands written and slot-annotated by hand by a native speaker |
| **Training** | LoRA (r=16, α=32) on Qwen3-0.6B/1.7B via mlx-lm and TRL; joint intent+BIO encoder baseline |
| **Evaluation** | One code path for MLX / PyTorch / API backends; exact match, slot F1, paired bootstrap CIs, leakage-cleaned subset |

## Repository layout

```
src/lahja/
  data/       MASSIVE parsing, Egyptian generation, rule filters, test-set tooling
  models/     prompts, inference backends (MLX / HF / encoder / Claude), encoder architecture
  train/      LoRA SFT (mlx-lm + TRL), encoder training, SFT data export
  eval/       metrics, runners (sync + batch), report, figures, latency benchmarks
configs/      schema, LoRA config, few-shot examples for generation
docs/         results, examples, model/dataset cards, Colab & Kaggle guides, CV bullets
results/      scores, per-item predictions, summary.md, figures/
tests/        37 tests (parsers, filters, metrics, staleness guard)
```

## Documentation

| | |
|---|---|
| [docs/results.md](docs/results.md) | Full results, figures, error analysis, limitations |
| [docs/examples.md](docs/examples.md) | Real inputs and outputs, including failures |
| [docs/huggingface.md](docs/huggingface.md) | Publishing the dataset and models to the Hub |
| [docs/larger_models.md](docs/larger_models.md) | Training bigger models on Apple silicon |
| [docs/colab.md](docs/colab.md) | GPU track (Colab and Kaggle notebooks) |
| [docs/annotation_guide.md](docs/annotation_guide.md) | How the human test set was written |
| [PLAN.md](PLAN.md) | Design, schedule, and a decisions log including what failed |

## Licence

Code **MIT** ([LICENSE](LICENSE)). Data derives from Amazon MASSIVE and is redistributed under
**CC BY 4.0** with attribution. Models inherit their base licences (Qwen3, CAMeLBERT: Apache 2.0).
