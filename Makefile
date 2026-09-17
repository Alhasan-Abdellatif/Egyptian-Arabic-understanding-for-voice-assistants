PY := uv run python
MASSIVE_URL := https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
EVAL_SETS := --eval-set data/processed/egy_test.jsonl --eval-set data/processed/massive_ar_test.jsonl

.PHONY: setup download data templates testset pilot submit status collect synth eval-a test lint

setup:           ## venv + dev deps
	uv sync --group dev --extra mlx --extra gpu  # on Colab: pip install -e ".[gpu]"
download:        ## MASSIVE 1.1 ar-SA (+ en-US for reference)
	mkdir -p data/raw && curl -sSL $(MASSIVE_URL) | tar -xz -C data/raw --include="*ar-SA.jsonl" --include="*en-US.jsonl" --include="*LICENSE*"
data:            ## parse MASSIVE into data/processed + configs/schema.json
	$(PY) -m lahja.data.prepare
templates:       ## CSVs for the hand-written Egyptian test set (never overwrites)
	$(PY) -m lahja.data.testset template
testset:         ## validate the CSVs -> data/processed/egy_{test,challenge}.jsonl
	$(PY) -m lahja.data.testset validate
pilot:           ## 100-seed synchronous generation run for prompt tuning
	$(PY) -m lahja.data.generate pilot --n 100
submit:          ## full generation via the Batches API
	$(PY) -m lahja.data.generate submit --n 4000
status:
	$(PY) -m lahja.data.generate status
collect:
	$(PY) -m lahja.data.generate collect
synth:           ## filter candidates -> egy_synth_{train,dev}.jsonl
	$(PY) -m lahja.data.build_synth
eval-a:          ## model A: Claude Opus 5 zero-shot (MASSIVE test subsampled to 1000)
	$(PY) -m lahja.eval.run_eval --backend anthropic --run-name A_opus5_zeroshot $(EVAL_SETS) --limit 1000
test:
	uv run pytest
lint:
	uv run ruff check src tests && uv run ruff format --check src tests

# ---- Day 2: training (Mac track = mlx-lm; GPU track = TRL on Colab) ----
MLX_MODEL ?= Qwen/Qwen3-0.6B
MLX_TAG ?= qwen3-06b
# Per-model knobs. Keep MLX_BATCH x MLX_ACCUM = 16 and MLX_ITERS = 1200 x MLX_ACCUM so every
# model gets the same 1200 optimizer updates over 16-example batches. See docs/larger_models.md.
MLX_BATCH ?= 4
MLX_ACCUM ?= 4
MLX_ITERS ?= 4800
MLX_MAXLEN ?= 256
MLX_CKPT ?=
MLX_TRAIN_FLAGS = --batch-size $(MLX_BATCH) --grad-accumulation-steps $(MLX_ACCUM) \
	--iters $(MLX_ITERS) --max-seq-length $(MLX_MAXLEN) $(if $(MLX_CKPT),--grad-checkpoint,)
.PHONY: sft eval-mlx-B  # pattern targets (train-mlx-%, eval-mlx-%) must not be .PHONY
sft:             ## export chat-format SFT mixes -> data/sft/{C,D}
	$(PY) -m lahja.data.build_synth >/dev/null && $(PY) -m lahja.train.export_sft
train-mlx-%:     ## LoRA SFT on this Mac, e.g. make train-mlx-D
	$(PY) -m mlx_lm lora -c configs/mlx_lora.yaml --model $(MLX_MODEL) --data data/sft/$* --adapter-path adapters/$*-$(MLX_TAG) $(MLX_TRAIN_FLAGS)
eval-mlx-B:      ## untrained small model, 5-shot, full schema prompt
	$(PY) -m lahja.eval.run_eval --backend mlx --model $(MLX_MODEL) --run-name B-$(MLX_TAG)-5shot --shots 5 $(EVAL_SETS) --limit 1000
eval-mlx-%:      ## fine-tuned adapter, short SFT prompt, e.g. make eval-mlx-D
	$(PY) -m lahja.eval.run_eval --backend mlx --model $(MLX_MODEL) --adapter adapters/$*-$(MLX_TAG) --prompt sft --run-name $*-$(MLX_TAG) $(EVAL_SETS) --limit 1000

# ---- model E: joint intent + BIO encoder ----
ENC_MODEL ?= CAMeL-Lab/bert-base-arabic-camelbert-da
ENC_TAG ?= camelbert-da
train-enc-%:     ## e.g. make train-enc-D
	$(PY) -m lahja.train.train_encoder --model $(ENC_MODEL) --mix $* --out models/E-$*-$(ENC_TAG)
eval-enc-%:      ## e.g. make eval-enc-D
	$(PY) -m lahja.eval.run_eval --backend encoder --model models/E-$*-$(ENC_TAG) --run-name E-$*-$(ENC_TAG) $(EVAL_SETS) --limit 1000

# ---- model A through the Batches API (50% cheaper, results within ~1h) ----
BATCH_MODEL ?= claude-sonnet-5
BATCH_RUN ?= A_sonnet5_zeroshot
HF_MODEL ?= Qwen/Qwen3-1.7B
HF_TAG ?= qwen3-17b
.PHONY: eval-a-batch eval-a-batch-status eval-a-batch-collect
eval-a-batch:    ## submit model A as one batch
	$(PY) -m lahja.eval.run_eval_batch submit --run-name $(BATCH_RUN) --model $(BATCH_MODEL) $(EVAL_SETS) --limit 300
eval-a-batch-status:
	$(PY) -m lahja.eval.run_eval_batch status --run-name $(BATCH_RUN)
eval-a-batch-collect:  ## download results and score them
	$(PY) -m lahja.eval.run_eval_batch collect --run-name $(BATCH_RUN)

quantize:        ## 4-bit local copy of MLX_MODEL -> models/$(MLX_TAG)-4bit (needed for 4B+ on 16GB)
	$(PY) -m mlx_lm convert --hf-path $(MLX_MODEL) -q --q-bits 4 --mlx-path models/$(MLX_TAG)-4bit

.PHONY: report figures examples
report:          ## regenerate results/summary.md from results/
	$(PY) -m lahja.eval.report --pairs D-$(MLX_TAG):C-$(MLX_TAG) E-D-$(ENC_TAG):E-C-$(ENC_TAG) \
		E-D-$(ENC_TAG):D-$(MLX_TAG) D-qwen3-17b:C-qwen3-17b D-qwen3-17b:D-$(MLX_TAG) \
		A_sonnet5_5shot:A_sonnet5_zeroshot D-qwen3-17b:A_sonnet5_5shot E-D-$(ENC_TAG):A_sonnet5_5shot

figures:         ## render report figures into results/figures/
	$(PY) -m lahja.eval.figures

# ---- latency / cost benchmarks (run them back-to-back on ONE machine) ----
BENCH_N ?= 50
.PHONY: bench-encoder bench-api  # pattern targets below must not be .PHONY
bench-encoder:   ## per-request latency of model E
	$(PY) -m lahja.eval.bench --backend encoder --model models/E-D-$(ENC_TAG) --name E-D-$(ENC_TAG) --n $(BENCH_N)
bench-mlx-%:     ## latency of a fine-tuned adapter on Apple silicon, e.g. make bench-mlx-D
	$(PY) -m lahja.eval.bench --backend mlx --model $(MLX_MODEL) --adapter adapters/$*-$(MLX_TAG) --prompt sft --name $*-$(MLX_TAG) --n $(BENCH_N)
bench-hf-%:      ## latency of a PEFT adapter (GPU or MPS), e.g. make bench-hf-D
	$(PY) -m lahja.eval.bench --backend hf --model $(HF_MODEL) --adapter adapters/$*-$(HF_TAG) --prompt sft --name $*-$(HF_TAG) --n $(BENCH_N)
bench-api:       ## latency + $/1k requests for the frontier model (synchronous endpoint)
	$(PY) -m lahja.eval.bench --backend anthropic --model $(BATCH_MODEL) --name A-$(BATCH_MODEL) --n 30

examples:        ## regenerate docs/examples.md from real predictions
	$(PY) -m lahja.eval.examples
