"""Run resumable LoRA SFT against the vulnops heuristic dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_utils import (
    detect_device,
    latest_checkpoint,
    load_jsonl,
    preferred_torch_dtype,
    set_default_env,
    write_json,
)


class JsonlSFTDataset(Dataset):
    """Mask prompt tokens so only the completion contributes to the loss."""

    def __init__(self, records: List[Dict[str, object]], tokenizer, max_length: int):
        self.examples: List[Dict[str, List[int]]] = []
        for record in records:
            prompt = str(record["prompt"])
            completion = str(record["completion"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

            input_ids = (prompt_ids + completion_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
            attention_mask = [1] * len(input_ids)
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.examples[index]


class JsonlMetricLogger(TrainerCallback):
    """Append metrics during training so partial runs are still inspectable."""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.metrics_path = output_root / "metrics" / "train_metrics.jsonl"
        self.manifest_path = output_root / "run_manifest.json"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "global_step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            **{key: float(value) if isinstance(value, (int, float)) else value for key, value in logs.items()},
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        write_json(
            self.manifest_path,
            {
                "status": "training",
                "global_step": int(state.global_step),
                "epoch": float(state.epoch or 0.0),
                "best_model_checkpoint": state.best_model_checkpoint,
                "log_history_entries": len(state.log_history),
            },
        )


class AbortOnInvalidLoss(TrainerCallback):
    """Stop training early when the run becomes numerically invalid."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        for key in ("loss", "eval_loss", "grad_norm"):
            value = logs.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                control.should_training_stop = True
                break
        return control


def build_training_args(args, output_root: Path, use_cpu: bool) -> TrainingArguments:
    warmup_steps = max(1, int(args.warmup_ratio * args.estimated_train_steps))
    return TrainingArguments(
        output_dir=str(output_root / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        load_best_model_at_end=False,
        use_cpu=use_cpu,
        fp16=False,
        bf16=False,
        max_grad_norm=0.5,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-root", default="artifacts/lora_qwen3_4b")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fresh-start", action="store_true")
    args = parser.parse_args()

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install peft before running LoRA training.") from exc

    output_root = (ROOT / args.output_root).resolve()
    data_dir = output_root / "data"
    train_records = load_jsonl(data_dir / "train.jsonl")
    eval_records = load_jsonl(data_dir / "eval.jsonl")
    if not train_records or not eval_records:
        raise RuntimeError("Missing train/eval JSONL data. Run scripts/generate_sft_data.py first.")

    set_default_env(output_root)
    device = detect_device()
    use_cpu = device == "cpu"
    torch_dtype = torch.float32 if device == "mps" else preferred_torch_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if device in {"cuda", "mps"}:
        model.to(device)

    train_dataset = JsonlSFTDataset(train_records, tokenizer, args.max_length)
    eval_dataset = JsonlSFTDataset(eval_records, tokenizer, args.max_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    updates_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)),
    )
    args.estimated_train_steps = max(1, math.ceil(args.num_train_epochs * updates_per_epoch))
    training_args = build_training_args(args, output_root, use_cpu)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[JsonlMetricLogger(output_root), AbortOnInvalidLoss()],
    )

    checkpoint_dir = output_root / "checkpoints"
    resume_checkpoint = None if args.fresh_start else latest_checkpoint(checkpoint_dir)
    write_json(
        output_root / "run_manifest.json",
        {
            "status": "starting_training",
            "device": device,
            "model": args.model,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "estimated_train_steps": args.estimated_train_steps,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        },
    )

    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(str(output_root / "adapter"))
    tokenizer.save_pretrained(str(output_root / "adapter"))

    final_eval = trainer.evaluate(eval_dataset=eval_dataset)
    summary = {
        "status": "finished",
        "device": device,
        "train_loss": float(train_result.training_loss),
        "global_step": int(trainer.state.global_step),
        "eval_loss": float(final_eval["eval_loss"]) if math.isfinite(float(final_eval["eval_loss"])) else None,
        "adapter_dir": str(output_root / "adapter"),
    }
    write_json(output_root / "training_summary.json", summary)
    write_json(output_root / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
