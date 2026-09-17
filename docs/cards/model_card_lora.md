---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
language:
  - ar
tags:
  - lora
  - peft
  - arabic
  - egyptian-arabic
  - intent-classification
  - slot-filling
  - voice-assistant
---

# Qwen3-1.7B LoRA for Egyptian Arabic NLU

A LoRA adapter that turns an Egyptian Arabic voice command into a tool call: one intent plus its
slots, as JSON. Trained on [Amazon MASSIVE](https://github.com/alexa/massive) `ar-SA` **plus 6.5k
LLM-generated, rule-filtered Egyptian rewrites**.

```
Input : صحيني بكرة الساعة ستة الصبح
Output: {"intent": "alarm_set", "slots": {"date": "بكرة", "time": "ستة الصبح"}}
```

## Prompt format — required

The model was trained with this exact system prompt and no examples. Using a different prompt
degrades it badly:

```
You are the language-understanding module of a voice assistant for Arabic speakers. Reply with JSON only: {"intent": "<intent>", "slots": {"<slot_type>": "<value>"}}.
```

The user turn is the raw command, nothing else. Valid intent and slot names are in
`configs/schema.json` in the GitHub repo.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ('You are the language-understanding module of a voice assistant for Arabic speakers. '
          'Reply with JSON only: {"intent": "<intent>", "slots": {"<slot_type>": "<value>"}}.')

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "Alhasan/qwen3-1.7b-egyptian-arabic-lora")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "الجو عامل ايه في اسكندرية النهارده"}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                 enable_thinking=False)
out = model.generate(**tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device),
                     max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][-128:], skip_special_tokens=True))
```

## Results

Exact match = intent **and** every slot correct, on 200 native-written Egyptian commands:

| Model | Egyptian test | MASSIVE test (Saudi/MSA) |
|---|---:|---:|
| **This adapter (Qwen3-1.7B + LoRA)** | **0.590** | 0.583 |
| Qwen3-0.6B + LoRA | 0.510 | 0.539 |
| Claude Sonnet 5, 5-shot | 0.435 | 0.450 |
| Qwen3-0.6B untrained, 5-shot | 0.040 | 0.030 |

Fine-tuning beats the frontier model prompted with five examples by **+15.5 points**
(95% CI [+8.5, +22.5], paired bootstrap). JSON validity is **100%** with plain greedy decoding.

## Training

LoRA rank 16, alpha 32, dropout 0.05, all attention and MLP projections; 1,200 optimizer steps at
batch 16, lr 2e-4 cosine, max length 256, loss on the assistant JSON only (completion-only).
Trained with TRL `SFTTrainer` on a single GPU (~1 h).

## Limitations

- **Egyptian dialect only.** Other dialects were not trained or evaluated.
- Bounded by MASSIVE's domain (smart-speaker commands) and by its arbitrary span conventions.
- On this task a **110M dialect-BERT encoder matches it** (0.625) at 25× lower latency — if you
  need only these 60 intents, that is the cheaper deployment. See the GitHub repo's results.

Full methodology, significance testing and figures: **[GitHub repo](https://github.com/Alhasan-Abdellatif/Egyptian-Arabic-understanding-for-voice-assistants)**
