# Training larger models

The comparison that matters (C vs. D) must keep the **same optimizer budget** for every model:
1,200 updates over 16-example batches. Only *how* those 16 examples are accumulated changes.

```
MLX_BATCH x MLX_ACCUM = 16          MLX_ITERS = 1200 x MLX_ACCUM
```

## What drives memory

| Term | Size | Notes |
|---|---|---|
| Weights | 2 bytes × params (bf16), 0.5 bytes (4-bit) | the base model is frozen, but still resident |
| Logits | `batch × seq × vocab × 4 bytes`, roughly ×3 with gradients | **vocabulary size dominates** |
| Activations | ∝ `layers × hidden × batch × seq` | `MLX_CKPT=1` trades ~25% speed to cut these |
| LoRA + optimizer | ~10M params × 12 bytes ≈ 0.1 GB | negligible |

This is why Qwen3.5-0.8B (248,320-word vocabulary) needed more memory than Qwen3-0.6B
(151,936) despite being only slightly larger, and why ALLaM (64,000) is unusually cheap per
parameter.

## Measured on this 16GB M4 (GPU budget ≈ 12.7 GB)

| Model | Vocab | Settings | Peak | Speed | 1,200 updates |
|---|---|---|---|---|---|
| Qwen3-0.6B | 152k | batch 4 × 4, no checkpointing | 5.0 GB | 1.3–1.8 it/s | ~50 min |
| Qwen3.5-0.8B | 248k | batch 4 × 4, checkpointing | 6.7 GB | 0.20 it/s | ~7 h (rejected) |
| Qwen3.5-0.8B | 248k | batch 4 × 4, no checkpointing | OOM | — | — |

## Suggested settings for larger models

Estimates below the measured rows; verify the first 100 steps before walking away.

| Model | Params / vocab | Command settings | Expected peak | Verdict |
|---|---|---|---|---|
| **Qwen3-1.7B** | 1.7B / 152k | `MLX_BATCH=2 MLX_ACCUM=8 MLX_ITERS=9600 MLX_CKPT=1` | ~8–9 GB | **best next step on the Mac**; ~2.5–3 h per run |
| Qwen3-4B (4-bit) | 4B / 152k | `make quantize` first, then `MLX_BATCH=2 MLX_ACCUM=8 MLX_ITERS=9600 MLX_CKPT=1` | ~8–10 GB | fits, but likely 8–10 h per run → prefer Colab |
| Qwen3-4B (bf16) | 4B / 152k | — | >12.7 GB | won't fit |
| Qwen3.5-2B | 2B / 248k | `MLX_BATCH=2 MLX_ACCUM=8 MLX_ITERS=9600 MLX_CKPT=1` | ~9–11 GB | avoid: the qwen3_5 architecture trains ~9× slower here |
| ALLaM-7B-Instruct (4-bit) | 7B / 64k | `make quantize`, then `MLX_BATCH=1 MLX_ACCUM=16 MLX_ITERS=19200 MLX_CKPT=1` | ~7–9 GB | Arabic-specialized (Saudi); too slow on the Mac → Colab |
| Fanar-1-9B-Instruct | 9B / 128k | — | — | Arabic-specialized (Qatari); Colab/A100 only |

Jais models are gated on the Hub and need a license click before download.

## Commands

```bash
# Larger model, same optimizer budget, on the Mac
make train-mlx-D MLX_MODEL=Qwen/Qwen3-1.7B MLX_TAG=qwen3-17b \
     MLX_BATCH=2 MLX_ACCUM=8 MLX_ITERS=9600 MLX_CKPT=1 2>&1 | tee logs/train-mlx-D-17b.log
make eval-mlx-D MLX_MODEL=Qwen/Qwen3-1.7B MLX_TAG=qwen3-17b

# 4-bit first for anything >= 4B
make quantize MLX_MODEL=Qwen/Qwen3-4B MLX_TAG=qwen3-4b
make train-mlx-D MLX_MODEL=models/qwen3-4b-4bit MLX_TAG=qwen3-4b-4bit \
     MLX_BATCH=2 MLX_ACCUM=8 MLX_ITERS=9600 MLX_CKPT=1

# GPU track (Colab), same budget
python -m lahja.train.sft_lora --model Qwen/Qwen3-4B --mix D --out adapters/D-qwen3-4b
```

Adapters and results are tagged by `MLX_TAG`, so runs never overwrite each other and the
evaluator's fingerprint check re-predicts automatically when weights change.

## Before starting a long run

- Free memory: quit other GPU-heavy apps; `MLX_CKPT=1` if the peak is near 10 GB.
- Watch the first 100–200 steps. Training loss should fall below ~0.5 with no jump above 1.0.
  A spike means the learning rate is too high for that model: halve it with `--learning-rate`.
- Capture the log: `... 2>&1 | tee logs/<name>.log`.
