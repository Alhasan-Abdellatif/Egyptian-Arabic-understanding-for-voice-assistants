PY := uv run python
MASSIVE_URL := https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
EVAL_SETS := --eval-set data/processed/egy_test.jsonl --eval-set data/processed/egy_challenge.jsonl --eval-set data/processed/massive_ar_test.jsonl

.PHONY: setup download data templates testset pilot submit status collect synth eval-a test lint

setup:           ## venv + dev deps
	uv sync --group dev
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
