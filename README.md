# Lahja (لهجة): Egyptian-Arabic voice-assistant understanding

Fine-tuning small LLMs to turn Egyptian Arabic commands into structured tool calls
(intent + slots), compared against a frontier LLM and a dialect-pretrained BERT encoder,
and run on-device with MLX.

> Work in progress. See [PLAN.md](PLAN.md) for the design and schedule.

## Quickstart

```bash
make setup      # uv venv + dev deps
make download   # MASSIVE 1.1 (ar-SA) from Amazon S3
make data       # parse into data/processed/*.jsonl + configs/schema.json
make test
```

## Data

- **MASSIVE `ar-SA`** (CC BY 4.0): 60 intents, 55 slot types. Mostly Saudi/Gulf colloquial and MSA.
- **Egyptian synthetic** (Claude rewrites of MASSIVE train seeds, rule-filtered).
- **Egyptian human test set**: written by a native Egyptian speaker from MASSIVE *test* items,
  plus a challenge set (code-switching, Arabizi, digits, Egyptian time expressions).
