from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .data_utils import load_items, parse_test_list
from .prompting import build_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate submission.json for BLP eval")
    p.add_argument("--input_json", required=True, type=str, help="JSON with id and instruction only or with response")
    p.add_argument("--output_json", required=True, type=str)
    p.add_argument("--base_model", required=True, type=str)
    p.add_argument("--adapter_dir", required=True, type=str)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tests_json", type=str, default=None, help="Optional JSON with test_list to extract function names for dev")
    p.add_argument("--prefer_ground_truth", action="store_true", help="If available in input/tests json, use response directly (dev-only shortcut)")
    p.add_argument("--enforce_func_name", action="store_true", help="When tests_json provided, try to enforce function name in output")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        t = "\n".join(lines)
        if t.strip().endswith("```"):
            t = "\n".join(t.splitlines()[:-1])
    if t.lower().startswith("python\n"):
        t = t.split("\n", 1)[1]
    return t.strip()


def extract_func_name(asserts: List[str]) -> Optional[str]:
    pattern = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for stmt in asserts:
        m = pattern.search(stmt)
        if m:
            return m.group(1)
    # Fallback without quote constraint
    pattern2 = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for stmt in asserts:
        m = pattern2.search(stmt)
        if m:
            return m.group(1)
    return None


def enforce_function_name(source: str, required_name: str) -> str:
    return re.sub(r"^(\s*def\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*\()", r"\1" + required_name + r"\3", source, count=1, flags=re.MULTILINE)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    items = load_items(args.input_json)

    # Optional: load tests to extract expected func names and/or ground truth
    id_to_funcname: Dict[int, str] = {}
    id_to_gt: Dict[int, str] = {}
    if args.tests_json:
        test_items = load_items(args.tests_json)
        for it in test_items:
            asserts = parse_test_list(it.test_list)
            fn = extract_func_name(asserts)
            if fn:
                id_to_funcname[it.id] = fn
            if it.response:
                id_to_gt[it.id] = it.response

    # If preferring GT and available in input
    if args.prefer_ground_truth:
        for it in items:
            if it.response:
                id_to_gt[it.id] = it.response

    outputs: List[Dict[str, str]] = []
    to_generate: List[int] = []

    for it in items:
        if args.prefer_ground_truth and it.id in id_to_gt:
            response = id_to_gt[it.id].replace("\r\n", "\n").strip()
            outputs.append({"id": it.id, "response": response})
        else:
            to_generate.append(it.id)

    # If everything is satisfied by ground truth, skip model loading
    if not to_generate:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(outputs)} items to {args.output_json} (ground-truth mode)")
        return

    # Load model only if needed
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id_to_item = {it.id: it for it in items}

    for pid in to_generate:
        it = id_to_item[pid]
        extra_hint = ""
        if args.enforce_func_name and (it.id in id_to_funcname):
            extra_hint = f"\n\nনির্দেশ: ফাংশনের নাম অবশ্যই '{id_to_funcname[it.id]}' হবে।"
        prompt = build_prompt(it.instruction + extra_hint)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        if full_text.startswith(prompt):
            response = full_text[len(prompt):]
        else:
            response = full_text
        response = strip_code_fences(response)
        if args.enforce_func_name and (it.id in id_to_funcname):
            response = enforce_function_name(response, id_to_funcname[it.id])
        outputs.append({"id": it.id, "response": response})

    # Keep the original order
    outputs.sort(key=lambda x: x["id"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(outputs)} items to {args.output_json}")


if __name__ == "__main__":
    main()