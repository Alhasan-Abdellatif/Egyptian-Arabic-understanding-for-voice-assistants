"""LoRA SFT with TRL on a CUDA GPU (Colab). Same data and step budget as the mlx-lm track.

    python -m lahja.train.sft_lora --model Qwen/Qwen3.5-2B --mix D --out adapters/D-qwen35-2b

Chat records are converted to TRL's prompt-completion format, so the loss covers only the
assistant JSON (completion_only_loss), matching `mask_prompt` in the mlx-lm config.
"""

from __future__ import annotations

import argparse

from lahja import DATA


def to_prompt_completion(row: dict) -> dict:
    return {
        "prompt": row["messages"][:-1],
        "completion": row["messages"][-1:],
        # Qwen3-style templates: train with the same empty think block used at inference.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mix", required=True, choices=["C", "D"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Colab's free T4 (Turing) has no bfloat16; fall back to fp16 there.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    sft_dir = DATA / "sft" / args.mix
    ds = load_dataset(
        "json",
        data_files={"train": str(sft_dir / "train.jsonl"), "valid": str(sft_dir / "valid.jsonl")},
    ).map(to_prompt_completion, remove_columns=["messages"])

    config = SFTConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_length=256,
        completion_only_loss=True,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=0,
    )
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)  # best checkpoint (lowest eval loss) -> adapter dir


if __name__ == "__main__":
    main()
