---
license: cc-by-4.0
language:
  - ar
task_categories:
  - text2text-generation
  - token-classification
tags:
  - arabic
  - egyptian-arabic
  - dialect
  - intent-classification
  - slot-filling
  - voice-assistant
  - synthetic-data
size_categories:
  - 1K<n<10K
---

# Egyptian Arabic voice-assistant NLU — dataset

Egyptian Arabic commands paired with **intent + slot** annotations, in the schema of
[Amazon MASSIVE](https://github.com/alexa/massive) (60 intents, 55 slot types).

Two parts, and the difference matters:

| File | Rows | Origin |
|---|---:|---|
| `egy_test.jsonl` | 200 | **Written and annotated by hand** by a native Egyptian speaker — the benchmark |
| `egy_synth_train.jsonl` | 6,509 | LLM-generated Egyptian rewrites of MASSIVE `ar-SA` training items |
| `egy_synth_dev.jsonl` | 730 | Same, held out by seed |

`egy_test.csv` is the raw annotation sheet, with each item's Saudi/MSA source next to the
Egyptian rewrite.

## Format

```json
{"id": "egy-main-massive-3161", "utt": "صحيني بكرة الساعة ستة الصبح",
 "intent": "alarm_set",
 "slots": [{"type": "date", "value": "بكرة", "start": 6, "end": 10},
           {"type": "time", "value": "ستة الصبح", "start": 17, "end": 26}],
 "variety": "egy", "source": "human", "split": "test", "scenario": "alarm"}
```

Slot values are verbatim spans with character offsets, following MASSIVE's conventions
(attached prepositions stay inside the span: `لستة الصبح`).

## How the synthetic data was made

MASSIVE `ar-SA` training items (mostly Saudi colloquial and MSA) were rewritten into Egyptian by
Claude Sonnet 5 through the Batches API, two variants per seed, in MASSIVE's bracket notation so
slot spans stay exact. 8,002 candidates → **7,239 kept** by rule-based filters — no LLM judge:

| Rejected because | Count |
|---|---:|
| Too close to the seed (not really rewritten) | 428 |
| Exact duplicate | 220 |
| Near duplicate | 60 |
| Slot types changed | 55 |

## Known limitations

- **The generated style is not the natural one.** Egyptian marker words appear in 60% of generated
  sentences but 32% of the human test set; English code-switching in 27% versus 10%. The generator
  writes "more Egyptian" than Egyptians do. Validate against the human split before trusting it.
- **19 of the 200 test items also appear verbatim in training data** — short commands like
  "امسح المنبه" that any two writers phrase identically. Scores on the clean 181 are reported
  alongside in the project's results.
- Domain is bounded by MASSIVE: smart-speaker commands, no long-form or multi-turn speech.

## Provenance and licence

Derived from **Amazon MASSIVE** (CC BY 4.0) and redistributed under **CC BY 4.0** with attribution.

```bibtex
@inproceedings{fitzgerald-etal-2023-massive,
  title = {{MASSIVE}: A 1{M}-Example Multilingual Natural Language Understanding Dataset},
  author = {FitzGerald, Jack and others},
  booktitle = {Proceedings of ACL},
  year = {2023}
}
```

Code, evaluation harness and full results: **[GitHub repo](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants)**
