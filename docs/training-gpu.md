# Running the SFT on Colab

**Fastest path:** upload [`notebooks/colab_sft.ipynb`](../notebooks/colab_sft.ipynb) to Colab
(File → Upload notebook) and run the cells top to bottom. The steps below explain what it does.

Same data, same prompt, same optimizer budget as the Mac track (1,200 updates × 16 examples),
so Colab results sit in the same results table. Only the model size changes.

## 0. Pick a GPU and a model

| Colab runtime | VRAM | bf16? | Recommended model | Time per run |
|---|---|---|---|---|
| T4 (free) | 16 GB | no (fp16 used automatically) | Qwen3-1.7B | ~1–1.5 h |
| L4 | 22 GB | yes | Qwen3-4B | ~45 min |
| A100 | 40 GB | yes | Qwen3-4B, or ALLaM-7B with 4-bit | ~30 min |

Qwen3-4B in fp16 does **not** fit a free T4; use Qwen3-1.7B there, or switch to an L4/A100.

### On Kaggle instead

Use [`notebooks/kaggle_sft.ipynb`](../notebooks/kaggle_sft.ipynb) — same steps, Kaggle paths and
upload flow already wired up. The Colab notebook works there too, with three changes:

- **Paths:** use `/kaggle/working` instead of `/content`.
- **Upload:** `google.colab.files` does not exist. Add the bundle as a Kaggle Dataset (or use the
  notebook's file upload), then `!tar xzf /kaggle/input/<dataset>/lahja-colab.tgz -C /kaggle/working`.
- **GPUs:** T4 ×2 or P100, none of which support bfloat16 — `sft_lora.py` detects this and uses
  fp16 automatically. Use Qwen3-1.7B; 4B needs 4-bit.

**Interpreter mismatch:** on Kaggle, `!pip` and `!python` can resolve to a different interpreter
than the notebook kernel, so an editable install becomes invisible
(`ModuleNotFoundError: No module named 'lahja'`). Install with `%pip` and run modules with
`!$PY -m ...` where `PY = sys.executable`, as the Kaggle notebook does.

**Variable substitution:** the notebook sets `MODEL`/`TAG` as environment variables and uses
`$MODEL` in commands. The `{MODEL}` form only expands inside an IPython cell — in a Kaggle
terminal or a `%%bash` cell it is passed through literally and fails with
`Repo id must use alphanumeric chars ... '{MODEL}'`.

## 1. Bundle the project (on the Mac)

The whole thing is ~1.1 MB, so a zip is simpler than GitHub:

```bash
cd ~/HASAN/Work/FT_project
COPYFILE_DISABLE=1 tar czf ~/Desktop/lahja-colab.tgz --exclude '__pycache__' --exclude '._*' \
    src pyproject.toml README.md configs data/sft \
    data/processed/egy_test.jsonl data/processed/massive_ar_test.jsonl
```

Build the archive with `tar`, not Finder's "Compress" — a Finder zip carries AppleDouble
metadata files (`._sft_lora.py`), and `COPYFILE_DISABLE=1` plus the excludes keep them out of a
`tar` archive too. If `._`-prefixed files do reach the remote machine, delete them there:
`find . -name "._*" -delete`.

`README.md` must be in the bundle: `pyproject.toml` declares it, so `pip install -e .` fails with
`Preparing editable metadata (pyproject.toml) did not run successfully` without it.

Alternative: push the repo to GitHub (private is fine) and `git clone` it in Colab instead of
steps 2–3. Better if you expect several Colab sessions, and you'll want the repo public for
your CV anyway.

## 2. Set up the notebook

Runtime → Change runtime type → GPU, then confirm what you got:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Upload the bundle (drag into the file pane, or):

```python
from google.colab import files; files.upload()      # pick lahja-colab.tgz
```

```python
!mkdir -p /content/lahja && tar xzf lahja-colab.tgz -C /content/lahja
%cd /content/lahja
```

## 3. Install

Colab already has a working torch; don't let pip replace it.

```python
!pip install -q -e .                                  # the lahja package itself
!pip install -q -U "trl>=1.13" peft datasets accelerate
!pip uninstall -q -y torchao                          # see note below
```

**torchao:** Colab preinstalls an old version (0.10.0) and current peft aborts with
`Found an incompatible version of torchao ... only versions above 0.16.0 are supported`
as soon as it wraps a layer in LoRA. We never use torchao, so removing it is the cheapest fix;
`pip install -U torchao` also works but can pull a different torch build.

If the editable install fails anyway, skip it and put the source on the path instead:

```python
%env PYTHONPATH=/content/lahja/src
!pip install -q -U "trl>=1.13" peft datasets accelerate rapidfuzz tqdm pyyaml
```

## 4. Train

```python
!python -m lahja.train.sft_lora --model Qwen/Qwen3-4B --mix D --out adapters/D-qwen3-4b
!python -m lahja.train.sft_lora --model Qwen/Qwen3-4B --mix C --out adapters/C-qwen3-4b
```

Defaults match the Mac track: 1,200 steps, batch 16, LoRA rank 16 (alpha 32), learning rate
2e-4, loss on the assistant JSON only. Precision is chosen automatically (bf16 where supported,
fp16 on T4). Useful overrides: `--batch-size 8`, `--max-steps`, `--lr`, `--rank`.

**Watch the first 100 steps.** Loss should fall below ~0.5 with no jump above 1.0. A spike means
the learning rate is too high for that model — rerun with `--lr 1e-4`. (That's exactly how the
first Mac runs were lost.)

## 5. Evaluate (on the GPU, same code as everywhere else)

```python
!python -m lahja.eval.run_eval --backend hf --model Qwen/Qwen3-4B \
    --adapter adapters/D-qwen3-4b --prompt sft --run-name D-qwen3-4b \
    --eval-set data/processed/egy_test.jsonl \
    --eval-set data/processed/massive_ar_test.jsonl --limit 300
```

Repeat with `--adapter adapters/C-qwen3-4b --run-name C-qwen3-4b`.

## 6. Bring the results home

```python
!tar czf /content/lahja-results.tgz results adapters
from google.colab import files; files.download('/content/lahja-results.tgz')
```

Then on the Mac, from the project root: `tar xzf ~/Downloads/lahja-results.tgz`. The results
folders are tagged per model, so nothing overwrites the 0.6B runs.

Adapters for a 4B model are ~100–200 MB. To keep them beyond the session, either download them
or push to the Hub (`huggingface-cli login`, then `huggingface-cli upload <repo> adapters/D-qwen3-4b`).

## Gotchas

- **Sessions die.** Free Colab disconnects when idle (~90 min) and caps at ~12 h. Save adapters
  to Drive or download them as soon as a run finishes.
- **Re-installs on reconnect.** A new session starts empty; steps 2–3 must be repeated.
- **The evaluator caches predictions** keyed by a fingerprint of the weights, so re-running after
  a retrain re-predicts automatically. `--fresh` forces it.
- **4-bit (QLoRA)** is not wired up in `sft_lora.py`. If you want ALLaM-7B or a 9B model on a
  single GPU, that needs `bitsandbytes` plus a `BitsAndBytesConfig` — ask and I'll add it.
