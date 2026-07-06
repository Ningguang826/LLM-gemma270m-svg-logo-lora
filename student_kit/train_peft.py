"""Transformers + PEFT 备用 LoRA 训练脚本。

优先训练方式仍是 AI Studio 上的 ms-swift；本脚本用于在本地保持训练可复现，
并明确实现“只在 assistant SVG 部分计算 loss”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from student_kit.prompt_utils import build_prompt as build_plain_prompt_shared


DEFAULT_TRAIN_PATH = Path("logo-detailed-prompt/train.jsonl")
DEFAULT_VALID_PATH = Path("logo-detailed-prompt/valid.jsonl")


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype, use_fp16, use_bf16 = resolve_precision(args.precision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        # 8GB 显存环境下优先保留训练可运行性；关闭 cache 避免 checkpointing 冲突。
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.train_path)
    if args.max_target_chars > 0:
        # 小模型更容易先学会短而完整的 SVG；这是“SVG 长度上限”实验的一部分。
        train_rows = [
            row for row in train_rows
            if len(get_message(row, "assistant")) <= args.max_target_chars
        ]
        if not train_rows:
            raise ValueError("--max-target-chars 过滤后没有训练样本")
    if args.train_fraction < 1.0:
        keep = max(1, int(len(train_rows) * args.train_fraction))
        train_rows = train_rows[:keep]
    valid_rows = read_jsonl(args.valid_path)

    train_dataset = LogoSvgDataset(train_rows, tokenizer, args.max_length)
    valid_dataset = LogoSvgDataset(valid_rows, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        seed=args.seed,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        fp16=use_fp16,
        bf16=use_bf16,
        max_steps=args.max_steps,
        logging_first_step=True,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_batch,
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./gemma3-270m")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/peft-gemma3-270m-logo"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("adapter"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--max-target-chars", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    return parser.parse_args()


def resolve_precision(precision: str) -> tuple[torch.dtype, bool, bool]:
    if not torch.cuda.is_available() or precision == "fp32":
        return torch.float32, False, False
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False, True
    if precision == "fp16" or precision == "auto":
        return torch.float16, True, False
    return torch.float32, False, False


class LogoSvgDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        messages = row["messages"]
        prompt = build_plain_prompt(self.tokenizer, messages)
        target = get_message(row, "assistant").strip()
        eos = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target + eos, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get_message(row: dict[str, Any], role: str) -> str:
    for message in row["messages"]:
        if message["role"] == role:
            return message["content"]
    return ""


def build_plain_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """已迁移到 student_kit.prompt_utils.build_prompt，此处保留转发以兼容旧引用。"""
    return build_plain_prompt_shared(tokenizer, messages)


def collate_batch(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [0] * pad_len)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
        batch["labels"].append(feature["labels"] + [-100] * pad_len)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


if __name__ == "__main__":
    main()
