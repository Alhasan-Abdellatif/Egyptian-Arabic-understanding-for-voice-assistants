# Egyptian Arabic understanding for voice assistants

[![ci](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants/actions/workflows/ci.yml/badge.svg)](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants/actions/workflows/ci.yml)

Turn an Egyptian Arabic voice command into a structured tool call — one intent plus its slots.

```
صحيني بكرة الساعة ستة الصبح
    → {"intent": "alarm_set", "slots": {"date": "بكرة", "time": "ستة الصبح"}}
```

Most Arabic assistant data is Modern Standard or Gulf Arabic, but Egyptians speak Egyptian. This
project builds the missing training data, fine-tunes models on it, and measures what actually
helps. 60 intents, 55 slot types, **100% valid JSON** output with plain greedy decoding.

## Install

```bash
git clone https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants
cd Egyptian-Arabic-understanding-for-voice-assistants
uv sync --extra gpu          # torch + transformers; works on CPU, GPU or Apple silicon
```

With pip instead: `pip install -e ".[gpu]"`. On Apple silicon add `--extra mlx` for fast local
inference of the LoRA model.

## Use it

Weights download from the Hugging Face Hub on first run and are cached.

```bash
python -m lahja.predict "صحيني بكرة الساعة ستة الصبح"
python -m lahja.predict --model lora "الجو عامل ايه في اسكندرية النهارده"
echo "شغل اغاني لعمرو دياب" | python -m lahja.predict --compact
```

```json
{"text": "صحيني بكرة الساعة ستة الصبح", "intent": "alarm_set",
 "slots": {"date": "بكرة", "time": "ستة الصبح"}}
```

From Python:

```python
from lahja.predict import Predictor

nlu = Predictor()                      # 110M encoder: ~20 ms/command, runs on a laptop CPU
print(nlu(["امسح المنبه", "ضيف البروكلي لقائمة البقالة"]))
```

`--model lora` swaps in the fine-tuned Qwen3-1.7B: slightly different strengths, much larger.
`--weights <dir>` uses your own checkpoint. More examples, including failures:
[docs/examples.md](docs/examples.md).

## Models and data

| Artifact | What it is | Size |
|---|---|---:|
| [`Alhasan/egyptian-nlu`](https://huggingface.co/datasets/Alhasan/egyptian-nlu) | 200 hand-written Egyptian test commands + 7.2k filtered synthetic training rows | 3 MB |
| [`Alhasan/camelbert-egyptian-arabic-nlu`](https://huggingface.co/Alhasan/camelbert-egyptian-arabic-nlu) | 110M joint intent + slot tagger — the default, best scoring | 417 MB |
| [`Alhasan/qwen3-1.7b-egyptian-arabic-lora`](https://huggingface.co/Alhasan/qwen3-1.7b-egyptian-arabic-lora) | LoRA adapter for Qwen3-1.7B | 70 MB |

## How well it works

Exact match means intent **and** every slot correct, on 200 Egyptian commands written by a native
speaker. Full report, figures and error analysis: **[docs/results.md](docs/results.md)**.

| Model | Egyptian test | MASSIVE test (Saudi/MSA) | Latency |
|---|---:|---:|---:|
| **CAMeLBERT 110M** (default) | **0.625** | 0.586 | 20 ms |
| Qwen3-1.7B + LoRA | 0.590 | 0.583 | — |
| Qwen3-0.6B + LoRA | 0.510 | 0.539 | 550 ms |
| Claude Sonnet 5, 5-shot | 0.435 | 0.450 | ~1 s (API) |
| Qwen3-0.6B untrained, 5-shot | 0.040 | 0.030 | — |

Fine-tuning beats prompting a frontier model by **+15.5 points** (95% CI [+8.5, +22.5], paired
bootstrap): its errors are the dataset's labelling conventions, not Arabic comprehension, and five
examples don't fix that. Synthetic Egyptian data gives the encoder a significant **+6.0**; the
Qwen3 models gain nothing measurable overall, but **~+6 on dialect-marked sentences**.

## Reproduce it

```bash
make setup && make download && make data     # MASSIVE ar-SA -> data/processed
make pilot && make submit && make collect    # generate Egyptian rewrites (Claude, Batches API)
make synth && make sft                       # filter -> data/sft/{C,D}
make train-mlx-D && make train-enc-D         # train
make eval-mlx-D && make eval-enc-D
make report && make figures                  # -> results/summary.md, results/figures/
```

Training on a GPU: [docs/training-gpu.md](docs/training-gpu.md) (Colab and Kaggle notebooks
included). On Apple silicon: [docs/training-apple-silicon.md](docs/training-apple-silicon.md).

## How it works

| Stage | What happens |
|---|---|
| **Data** | [Amazon MASSIVE](https://github.com/alexa/massive) `ar-SA` (Saudi/MSA) parsed into intent + slot spans |
| **Generation** | Claude Sonnet 5 rewrites 4k seeds into Egyptian, two variants each, in MASSIVE bracket notation |
| **Filtering** | Rule-based only — slot preservation, copy and near-duplicate detection. 8,002 → **7,239** kept |
| **Benchmark** | 200 Egyptian commands written and slot-annotated by hand ([guide](docs/annotation_guide.md)) |
| **Training** | LoRA (r=16, α=32) on Qwen3 via mlx-lm and TRL; joint intent + BIO encoder baseline |
| **Evaluation** | One code path for every backend; exact match, slot F1, paired bootstrap CIs, leakage-cleaned subset |

## Project layout

```
src/lahja/
  predict.py    inference entry point (CLI + Predictor class)
  data/         MASSIVE parsing, Egyptian generation, filters, test-set tooling
  models/       prompts, backends (encoder / MLX / transformers / Claude), encoder architecture
  train/        LoRA SFT (mlx-lm + TRL), encoder training, SFT data export
  eval/         metrics, runners, report, figures, latency benchmarks
configs/        label schema, LoRA config, few-shot examples for generation
results/        scores, per-item predictions, summary.md, figures/
docs/           results, examples, training guides, publishing
tests/          37 tests
```

## Licence

Code **MIT** ([LICENSE](LICENSE)). Data derives from Amazon MASSIVE and is redistributed under
**CC BY 4.0** with attribution. Models inherit their base licences (Qwen3, CAMeLBERT: Apache 2.0).
