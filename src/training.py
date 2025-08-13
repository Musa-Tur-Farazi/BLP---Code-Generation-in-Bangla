from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import inspect

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType

from .data_utils import load_items
from .prompting import build_prompt


@dataclass
class TrainSample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def _tokenize_example(tokenizer, prompt: str, response: str, max_seq_len: int) -> TrainSample:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # Normalize line endings
    response_norm = response.replace("\r\n", "\n").strip() + "\n"
    target_ids = tokenizer(response_norm, add_special_tokens=False)["input_ids"]

    # Truncate from the left if needed, but preserve full response when possible
    max_input = max_seq_len - 1  # leave room for EOS
    total = len(prompt_ids) + len(target_ids) + 1
    if total > max_input:
        overflow = total - max_input
        # Prefer trimming prompt first
        trim_prompt = min(overflow, max(0, len(prompt_ids) - 8))
        prompt_ids = prompt_ids[trim_prompt:]
        overflow -= trim_prompt
        if overflow > 0 and len(target_ids) > overflow:
            target_ids = target_ids[:-overflow]

    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)

    # Mask out prompt tokens from loss with -100
    labels = ([-100] * len(prompt_ids)) + target_ids + [tokenizer.eos_token_id]

    return TrainSample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path: str, max_seq_len: int) -> None:
        super().__init__()
        items = load_items(data_path)
        self.samples: List[TrainSample] = []
        for it in items:
            if not it.response:
                # skip items without targets in training
                continue
            prompt = build_prompt(it.instruction)
            sample = _tokenize_example(tokenizer, prompt, it.response, max_seq_len)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(s.attention_mask, dtype=torch.long),
            "labels": torch.tensor(s.labels, dtype=torch.long),
        }


class DataCollatorForCausal:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            l = len(f["input_ids"]) 
            pad = max_len - l
            input_ids.append(torch.nn.functional.pad(f["input_ids"], (0, pad), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(f["attention_mask"], (0, pad), value=0))
            labels.append(torch.nn.functional.pad(f["labels"], (0, pad), value=-100))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "c_attn",
                "proj",
            ]
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for BLP Bangla->Code")
    p.add_argument("--train_json", required=True, type=str)
    p.add_argument("--eval_json", required=False, type=str, default=None)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_cfg)

    train_ds = SupervisedDataset(tokenizer, args.train_json, args.max_seq_len)
    eval_ds = SupervisedDataset(tokenizer, args.eval_json, args.max_seq_len) if args.eval_json else None
    collator = DataCollatorForCausal(tokenizer)

    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
    )

    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if eval_ds is not None:
        if "evaluation_strategy" in sig_params:
            ta_kwargs["evaluation_strategy"] = "steps"
            if "eval_steps" in sig_params:
                ta_kwargs["eval_steps"] = args.eval_steps
        elif "evaluate_during_training" in sig_params:
            ta_kwargs["evaluate_during_training"] = True
            if "eval_steps" in sig_params:
                ta_kwargs["eval_steps"] = args.eval_steps

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save adapter and tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Print final perplexity if eval was run
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        eval_loss = last.get("eval_loss")
        if eval_loss is not None and eval_loss == eval_loss:  # not NaN
            ppl = math.exp(eval_loss)
            print(f"Final eval ppl: {ppl:.3f}")


if __name__ == "__main__":
    main()