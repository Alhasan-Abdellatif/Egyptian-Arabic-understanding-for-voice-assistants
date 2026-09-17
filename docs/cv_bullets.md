# CV bullets

Pick one block. Numbers are from [results.md](results.md); keep them accurate if you edit.

## One line (tight CV)

> **Egyptian-Arabic voice-assistant NLU** — LoRA fine-tuned Qwen3 (0.6B/1.7B) to convert Egyptian
> dialect commands into structured tool calls (intent + slots); built an LLM-generated,
> rule-filtered 7.2k-example dialect corpus and a native-written benchmark, reaching **0.59 exact
> match vs 0.44 for Claude Sonnet 5 few-shot (+15 pts, significant)** with 100% valid JSON output.
> [GitHub]

## Two lines (recommended)

> **Egyptian-Arabic Voice-Assistant NLU** — Fine-tuned Qwen3-0.6B/1.7B with **LoRA (PEFT,
> TRL, MLX)** to map Egyptian dialect commands to tool calls (60 intents, 55 slot types),
> generating a 7.2k-example dialect corpus with Claude (batch API, rule-based filtering) and a
> 200-command native-written test set.
> Fine-tuned models beat **Claude Sonnet 5 few-shot by +15 points exact match** (0.590 vs 0.435,
> paired bootstrap, p < 0.01); a 110M dialect-BERT baseline matched the 1.7B LLM at **25× lower
> latency (20 ms on-device, 4-bit MLX on Apple silicon)**. [GitHub] [HF]

## Detailed (portfolio / LinkedIn projects section)

> **Egyptian-Arabic understanding for voice assistants**
> - **Fine-tuned Qwen3-0.6B and Qwen3-1.7B with LoRA** (rank 16, completion-only loss) to turn
>   Egyptian Arabic commands into structured tool calls — 60 intents, 55 slot types, **100% valid
>   JSON** with no constrained decoding.
> - **Built the training data**: generated 8k Egyptian rewrites of MASSIVE with Claude Sonnet 5
>   (Batches API, few-shot prompting), kept 7.2k after rule-based filtering (slot preservation,
>   near-duplicate and copy detection) — no LLM judge, fully reproducible.
> - **Built the benchmark**: 200 Egyptian commands written and slot-annotated by hand (native
>   speaker), with train/test leakage auditing and a leakage-cleaned score reported alongside.
> - **Results**: 0.590 exact match vs **0.435 for Claude Sonnet 5 with 5 examples (+15.5 pts,
>   95% CI [+8.5, +22.5])**; synthetic dialect data gave a **significant +6.0** to a 110M
>   dialect-BERT baseline, which matched the 1.7B LLM at **20 ms/query on-device** (MLX, Apple
>   silicon) versus ~1 s for the API.
> - **Engineering**: single evaluation path across MLX / PyTorch / API backends, paired bootstrap
>   significance testing, resumable batch inference, 37 unit tests, CI, and a reproducible
>   `make` pipeline from raw data to figures.

## Skills this project evidences

LoRA / PEFT · SFT · Qwen3 · MLX (Apple silicon) · Hugging Face (transformers, TRL, PEFT, Hub) ·
Claude API (Batches, structured outputs, prompt caching) · synthetic data generation & filtering ·
BERT encoders (joint intent classification + BIO slot tagging) · evaluation design, paired
bootstrap / McNemar significance testing · on-device quantization & latency benchmarking ·
Arabic NLP & dialect handling · Python, pytest, CI, Makefile pipelines
