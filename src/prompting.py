from __future__ import annotations

from dataclasses import dataclass


BANGla_PROMPT_TEMPLATE: str = (
    "নীচের নির্দেশনা অনুযায়ী শুধুমাত্র একটি বৈধ Python ফাংশনের সংজ্ঞা লিখুন। "
    "অতিরিক্ত কোনো টেক্সট, ব্যাখ্যা, বা কোডব্লক মার্কআপ ব্যবহার করবেন না।\n\n"
    "নির্দেশনা:\n{instruction}\n\n"
    "উত্তর:\n"
)


def build_prompt(instruction: str) -> str:
    cleaned_instruction: str = (instruction or "").strip()
    return BANGla_PROMPT_TEMPLATE.format(instruction=cleaned_instruction)


@dataclass(frozen=True)
class StopConfig:
    eos_token_texts: tuple[str, ...] = ("```",)
    max_new_tokens: int = 512


DEFAULT_STOP_CONFIG = StopConfig()