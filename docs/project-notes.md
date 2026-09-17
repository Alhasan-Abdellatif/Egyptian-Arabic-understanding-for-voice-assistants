# Egyptian-Arabic Voice Assistant Understanding — plan

A 3-day project: fine-tune a small LLM to turn Egyptian Arabic commands into structured tool calls (intent + slots), compare it against a frontier LLM and a BERT encoder, and run it on-device with MLX.

## 0. Decisions log

- **2026-09-15, Day 1 start:**
  - Claude API (Opus 5) for data generation and model A.
  - Training on Colab GPU, plus a Mac track: a ~0.5B model LoRA-trained locally with `mlx-lm`, reported as an extra row (C/D-0.5B-mlx).
  - Egyptian only; no LLM judge; speech optional.
- **Finding:** MASSIVE `ar-SA` is *not* mostly MSA. It is largely Saudi/Gulf colloquial (تكفى، أبغى، حقي) mixed with MSA, and numbers are always spelled out. Wherever this plan says "MSA", read "MASSIVE `ar-SA`". The research question becomes Saudi/MSA → Egyptian transfer.
- **Generation format:** rewrites are produced in MASSIVE bracket notation, so slot spans are exact by construction. The test set uses the same notation.
- **2026-09-16, synthetic data:** Sonnet 5 through the Batches API, 4,001 seeds × 2 variants → 8,002 candidates → 7,239 kept.
  - Rejected: 428 copies of the seed, 220 exact duplicates, 60 near duplicates, 55 with changed slot types.
  - Egyptian marker rate is 60% in the rewrites vs. 2.8% in the seeds. 27% of the rewrites contain Latin-script words (mostly in variant 2), which is more code-switching than natural speech.
- **2026-09-16, Day 2 setup:**
  - **Mac small LLM:** Qwen3-0.6B, which officially lists Egyptian Arabic among its languages. Its chat template gives identical training and inference prompts when thinking is disabled.
    - Qwen3.5-0.8B was tried first and rejected. Its 248k vocabulary caused an out-of-memory crash at batch 16 on the 16GB M4. At micro-batch 4 it trained at 0.196 steps/s (about 6.8 h per run), and without gradient checkpointing it ran out of memory.
    - Qwen3-0.6B at micro-batch 4 × 4 accumulation, checkpointing off: 1.82 steps/s, 4.3 GB peak, about 45 min per run.
  - **Encoder:** CAMeLBERT-DA.
  - **Short prompt for fine-tuned models:** they learn the label set from data, so their prompt drops the schema. The zero-shot and few-shot baselines keep the full-schema prompt.
  - **Fixed step budget:** C and D train for the same number of steps (1,200 × batch 16), so D doesn't simply see more examples. That makes a separate C+ run unnecessary.
- **2026-09-16, first LoRA runs diverged — discarded.** Training loss exploded in both runs (C at step 300: 0.92 → 9.36; D at step 500: 0.41 → 3.42). D partly recovered (val loss 0.165) but scored only 0.31 EM on held-out in-distribution data; C collapsed to constant predictions (3 intents, fixed slot values) and scored 0.00 EM.
  - **Cause:** `scale: 20.0` is mlx-lm's default for rank 8 with lr 1e-5, and its trainer has no gradient clipping. Pairing it with rank 16 and lr 2e-4 over all 28 layers was unstable.
  - **Fix:** scale 2.0 (alpha 32 / rank 16, same as the TRL config), lr 1e-4, warmup 100. Retrain C and D; treat every LLM number from the first runs as void.
- **2026-09-16, retrained D with the fixed config — stable.** No loss spikes; val loss 3.296 → 0.161 (step 800) → 0.090 (step 4800), vs 0.165 for the diverged run. Train 0.072 at the end, so no overfitting. Peak 5.0 GB, ~1.3 steps/s. Log: `logs/train-mlx-D.log`.
- **Test-set overlap:** 19 of 200 test sentences also appear verbatim in training data (short commands like "امسح المنبه"), and models score much higher on them (e.g. E-D: 0.79 vs 0.61). Report the clean 181-item subset alongside the full 200.

---

## 1. Objective

**Problem.** Assistant NLU data in Arabic is mostly Modern Standard Arabic (MSA), but Egyptians talk to their phones in Egyptian Arabic ("صحّيني بكرة", "شغّلي حاجة لعمرو دياب", "اعملي reminder"). Models trained on MSA lose accuracy on dialect input.

**Research question.**
> How much of the MSA → Egyptian gap can LLM-generated Egyptian data plus LoRA fine-tuning of a small (~1.5–4B) LLM close? Can it match a frontier LLM while running on-device, and does it actually beat a 110M dialect-pretrained BERT encoder?

**Hypotheses.**
- H1: A model trained on MSA only (C) is clearly worse on Egyptian than on MSA.
- H2: Adding filtered synthetic Egyptian data (D) closes most of that gap without hurting MSA.
- H3: A 4-bit quantized D keeps nearly all its accuracy and runs at interactive latency on an Apple M4.
- Open question: LLM (D) vs. encoder (E). Report whichever wins, and at what latency/size cost.

---

## 2. Task definition

This is **joint intent classification + slot filling** (the NLU step of Siri), framed as **generation** for the LLMs and as **classification + token tagging** for the encoder.

### LLM input / output (SFT pair)

```
[system]
You convert a user request into a tool call. Reply with JSON only.
Intents: alarm_set, alarm_query, play_music, weather_query, calendar_set, ...
Slots: time, date, artist_name, place_name, event_name, ...

[user]
صحّيني بكرة الساعة ٦ عشان الفجر

[assistant]  ← target; loss computed only on these tokens
{"intent": "alarm_set", "slots": {"date": "بكرة", "time": "الساعة ٦"}}
```

More examples:

| Input | Output |
|---|---|
| شغّلي حاجة لعمرو دياب | `{"intent": "play_music", "slots": {"artist_name": "عمرو دياب"}}` |
| الجو عامل ايه في اسكندرية النهارده؟ | `{"intent": "weather_query", "slots": {"place_name": "اسكندرية", "date": "النهارده"}}` |
| اعملي reminder اكلم ماما بالليل | `{"intent": "calendar_set", "slots": {"event_name": "اكلم ماما", "timeofday": "بالليل"}}` |

- Slot values are **verbatim spans** from the input (MASSIVE convention). They are not normalized, so there is no conversion to `06:00`.
- Intent/slot names are exactly MASSIVE's labels. Check the real label set on Day 1; the names above are illustrative.

### Encoder input / output (model E)

```
tokens:  صحّيني  بكرة    الساعة   ٦       عشان  الفجر
tags:    O       B-date  B-time   I-time  O     O
intent:  alarm_set
```

---

## 3. Data

### 3.1 Source: Amazon MASSIVE, `ar-SA` (CC BY 4.0)
- 60 intents, 55 slot types, 18 scenarios (alarm, weather, music, calendar, iot, ...).
- Splits: ~11.5k train / ~2k dev / ~3k test, mostly MSA.
- Annotation format: `annot_utt` = `صحيني [date : غدا] [time : الساعة السادسة]` → parse into (utterance, intent, spans).

### 3.2 Synthetic Egyptian training data (LLM-generated)
- **Seeds:** ~4k MASSIVE `ar-SA` train utterances, sampled stratified by intent.
- **Generation:** for each seed, a frontier LLM writes **2 Egyptian rewrites** (→ ~8k candidates). Claude Opus 5 (Batches API) gets the seed in MASSIVE bracket notation and returns rewrites in the same notation:
  ```json
  {"variants": ["صحيني [date : بكرة] الساعة [time : ستة الصبح]", "اظبطلي alarm [date : بكرة] على [time : ٦ الصبح]"]}
  ```
  - Keep the same intent and the same slot *types*. Slot *values* may change to Egyptian wording (غدا → بكرة).
  - Ask for natural variety: Egyptian particles (عايز، ازاي، فين، دلوقتي، مش، كده، هـ/بـ prefixes), some code-switching (English words), some Arabic-Indic digits.
  - Few-shot the prompt with ~10 rewrites you write yourself.
- **Rule-based filters (no LLM judge):**
  1. The bracket annotation parses (balanced brackets, non-empty slot values).
  2. **Slot preservation:** slot types (with counts) are identical to the seed's.
  3. **Not a copy:** after normalization, the rewrite is not identical or nearly identical to the MSA seed (char-level similarity threshold).
  4. **Dedup:** exact and near-duplicate removal across the whole set.
  5. *(Signal only, not a filter)* Egyptian-marker lexicon hit rate, reported in the dataset card.
- Log the rejection count per filter; it goes in the README.
- Hold out 10% of the synthetic set as **Egyptian dev** (for checkpoint selection only, never for final numbers).
- Cost: ~4k API calls with short outputs, a few USD.

### 3.3 Human-verified Egyptian test set (the key asset)
- **~250 examples, written by you** (native Egyptian), in two parts:
  - **Egyptian-MASSIVE (200):** sample 200 MASSIVE `ar-SA` **test** items stratified by intent, rewrite each in natural Egyptian, and re-mark the slot spans in your own sentence.
  - **Egyptian-Challenge (50):** harder or cultural cases mapped to existing MASSIVE intents: code-switching ("اعملي reminder", "شغّل الـ playlist"), Arabizi (a few), Arabic-Indic vs. Latin digits, Egyptian time expressions (بعد العصر، الفجر، كمان ساعة), local entities (places, artists, dishes).
- **Anti-leakage rules:**
  - Write these **before** looking at synthetic outputs. Never generate them with the LLM used for training data.
  - Seeds for synthetic data come from MASSIVE **train** only; test items come from MASSIVE **test** only.
- Tooling: a tiny CLI/Streamlit annotator (show MSA item → type Egyptian → mark spans) or a spreadsheet plus a validation script (spans must be substrings).
- Budget: ~3h of your time.

### 3.4 Final evaluation sets
| Set | Size | Purpose |
|---|---|---|
| MASSIVE `ar-SA` test (MSA) | ~3k | Did we keep MSA performance? |
| Egyptian-MASSIVE (human) | 200 | Main result |
| Egyptian-Challenge (human) | 50 | Robustness + error analysis |

---

## 4. Models / experiments

| ID | Model | Training | Purpose |
|---|---|---|---|
| A | Frontier LLM (Claude / GPT / Gemini) | none, zero-shot + tool schema | Upper reference |
| B | Small LLM (~1.5–4B instruct) | none, few-shot | Untrained baseline |
| C | Small LLM + LoRA | SFT on MASSIVE `ar-SA` only | Measures the dialect gap (H1) |
| D | Small LLM + LoRA | SFT on MASSIVE `ar-SA` + Egyptian synthetic | Main model (H2) |
| D-q4 | D merged → MLX 4-bit | — | On-device (H3) |
| E1 / E2 | CAMeLBERT-DA or MARBERT (~110M) | Full fine-tune on C data / D data | Encoder baseline |

- **Small LLM choice (Day 1, ~1h):** run B few-shot on 50 Egyptian-dev items for 2–3 candidates (e.g. Qwen3-1.7B, Qwen3-4B, Gemma-3-4B-it, or current equivalents) and pick the best Arabic performer that fits the time budget.
- **Why LoRA:** even "small" LLMs have billions of weights. A full fine-tune with Adam needs ~16 bytes/param (4B → ~64GB). LoRA trains <1% of the weights, fits a cheap GPU, trains fast, yields a small adapter, and preserves general Arabic ability.
- **SFT config (starting point):** TRL `SFTTrainer`, completion-only loss, LoRA r=16, α=32, dropout 0.05, all linear layers, lr 2e-4, 2 epochs, max length 512, bf16. Keep the same config for C and D.
- **Fairness caveat:** D sees more examples than C. If time allows, add **C+** (MSA upsampled to D's size) so the gain isn't just "more data".
- **Encoder (E):** a shared encoder with an intent head (CLS) plus a BIO token-classification head, 3–5 epochs, minutes on a GPU.
- **Inference:** greedy decoding, parse the JSON; an unparseable output counts as wrong on every metric.

---

## 5. Metrics

Reported per eval set:
- **Intent accuracy**
- **Slot micro-F1** (exact type + span match after light normalization: strip tatweel/diacritics, unify alef forms, applied identically to all models)
- **Exact match** (intent and full slot set correct)
- **JSON validity rate** (LLMs only)
- **Efficiency** (on the M4): p50/p95 latency per query, tokens/s, peak memory, model size on disk for B/C/D (fp16 where it fits) vs. D-q4 vs. E

**Headline table:**

| Model | MSA EM | EGY EM | EGY-Challenge EM | Latency p50 | Size |
|---|---|---|---|---|---|
| A | | | | (API) | — |
| B | | | | | |
| C | | | | | |
| D | | | | | |
| D-q4 | | | | | |
| E2 | | | | | |

**Error analysis (Day 2–3):** confusion between top intents; slot errors by type (time/date expressions are likely the hardest); code-switching failures; 10 annotated failure examples in the README.

---

## 6. Repo structure

```
FT_project/
├── PLAN.md
├── README.md
├── pyproject.toml            # uv/pip, ruff, pytest config
├── Makefile                  # make data / train / eval / bench / demo
├── configs/                  # yaml per experiment (B, C, D, E)
├── src/lahja/
│   ├── data/
│   │   ├── massive.py        # load + parse annot_utt → (utt, intent, spans)
│   │   ├── formats.py        # → chat SFT format, → BIO format
│   │   ├── generate.py       # Egyptian rewrite generation (async, cached, resumable)
│   │   └── filters.py        # JSON / slot-preservation / copy / dedup filters
│   ├── models/
│   │   ├── prompts.py        # system prompt + few-shot builder
│   │   ├── llm_infer.py      # HF + MLX + API backends behind one interface
│   │   └── encoder.py        # joint intent + BIO model
│   ├── train/
│   │   ├── sft_lora.py       # TRL SFT (runs on Modal)
│   │   └── train_encoder.py
│   ├── eval/
│   │   ├── metrics.py        # intent acc, slot F1, EM, JSON validity
│   │   ├── run_eval.py       # model × eval set → predictions.jsonl + scores.json
│   │   └── bench.py          # latency / memory on M4
│   └── annotate/app.py       # tiny test-set annotation tool
├── modal_app.py              # GPU training entrypoints
├── tests/                    # parser, formats, filters, metrics, JSON parsing
├── data/                     # gitignored; published to HF Hub
├── results/                  # scores + predictions per run (committed)
└── .github/workflows/ci.yml  # ruff + pytest
```

---

## 7. Schedule (3 days)

### Day 1: data + evaluation code
| Block | Task | Output |
|---|---|---|
| 1h | Repo skeleton, pyproject, CI, Modal setup | Green CI |
| 1.5h | MASSIVE loader + `annot_utt` parser + format converters (+ tests) | `massive_ar_{train,dev,test}.jsonl` |
| **3h (you)** | Write the Egyptian test set (200 + 50) + validation script | `egy_test.jsonl`, `egy_challenge.jsonl` |
| 1.5h | Generation pipeline + filters (+ tests); pilot on 100 seeds, you review ~30 by eye, tune prompt | Prompt v2 |
| 1h | Full generation run (~4k seeds) + filtering + stats | `egy_synth_{train,dev}.jsonl` |
| 1.5h | Metrics + eval runner (+ tests); run **A** | First results row |

*Order tip:* write the test set **before** reviewing the synthetic data (anti-leakage).

### Day 2: experiments
| Block | Task | Output |
|---|---|---|
| 1h | Candidate small LLM probe (B few-shot), pick model; full B eval | Row B |
| 2–3h | Train **C** and **D** on Modal (in parallel if possible); meanwhile train **E1/E2** | Adapters + encoders |
| 1.5h | Evaluate C, D, E1, E2 on all three sets | Rows C, D, E |
| 1.5h | Error analysis + (optional) C+ run | Notes for README |

### Day 3: on-device + write-up
| Block | Task | Output |
|---|---|---|
| 1.5h | Merge D → convert to MLX → 4-bit; eval D-q4; latency/memory bench for D, D-q4, E | Row D-q4 + efficiency columns |
| 1.5h | Demo (Streamlit/Gradio): type an Egyptian command → JSON → mock action; record GIF | `demo/` |
| 2h | README (motivation, method, data, results, error analysis, limitations), dataset + model cards, push to HF Hub | Public repo |
| 1h | Update CV, LinkedIn post | Apply |
| *(optional)* | Speech front end (section 8) | Extra row / demo |

---

## 8. Optional: speech front end

Only if Day 3 has slack.
- `mlx-whisper` (large-v3-turbo) on the M4 → transcript → D-q4 → JSON.
- Record ~30 of your own Egyptian test commands (phone mic).
- Report ASR WER, end-to-end EM (speech → JSON) vs. text EM on the same 30, and end-to-end latency.
- This adds a speech line to the CV and answers the "ASR/TTS" preferred qualification.

---

## 9. Cut list (if behind, in this order)

1. Speech (already optional)
2. C+ fairness run
3. Demo UI → replace with a CLI + asciinema/GIF
4. E1 (keep E2 only)
5. Egyptian-Challenge set → reduce to 25

**Never cut:** human Egyptian test set, C vs. D comparison, D-q4 on-device numbers, README with results.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Synthetic data is "MSA with a few words changed" | Few-shot with your own rewrites; copy filter; eyeball 30 samples in the pilot |
| Slot spans drift in rewrites | Slot-preservation filter (substring + same types) |
| Small LLM outputs invalid JSON | Completion-only SFT fixes this quickly; report validity rate; constrained decoding only if needed |
| Modal/GPU issues | Fall back to Colab; 1.5B model LoRA can also run (slowly) with MLX-LM on the M4 |
| Test set too small for tight conclusions | Report bootstrap 95% CIs on EM |
| MASSIVE Arabic label noise | Mention in limitations; spot-check during annotation |

---

## 11. Deliverables

- GitHub repo (tested, CI, reproducible via `make`)
- HF Hub: Egyptian synthetic dataset + human test set (with dataset card), LoRA adapter / MLX 4-bit model
- README with the headline table, error analysis, limitations
- Demo GIF
- CV bullet + LinkedIn post

**CV bullet template** (fill in the real numbers):
> Fine-tuned a *N*B LLM with LoRA (SFT) for Egyptian-Arabic voice-assistant tool calling (intent + slots). Built a filtered LLM data-generation pipeline (*N*k utterances) and a native-written Egyptian test set; raised Egyptian exact-match from *X*% (MSA-only) to *Y*%, reaching *Z*% of [frontier LLM], at *T* ms/query on-device (MLX 4-bit, Apple M4). Benchmarked against a dialect-pretrained BERT encoder.

---

## 12. Decisions needed before starting

1. LLM API for generation + model A: Claude, OpenAI, or Gemini (and is a key set up)?
2. GPU: Modal or Colab?
3. Hugging Face account for publishing data/models?
