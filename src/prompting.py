from __future__ import annotations

from dataclasses import dataclass


FEW_SHOT_EXAMPLES = [
    {
        "instruction": "দুটি পূর্ণসংখ্যার যোগফল নির্ণয় করো",
        "script": "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)"
    },
    {
        "instruction": "একটি সংখ্যার বর্গ নির্ণয় করো",
        "script": "import sys\nn = int(sys.stdin.read())\nprint(n * n)"
    },
    {
        "instruction": "একটি তালিকার সর্বোচ্চ মান খুঁজুন",
        "script": "import sys\narr = list(map(int, sys.stdin.read().split()))\nprint(max(arr))"
    },
    {
        "instruction": "একটি স্ট্রিং বিপরীত করুন",
        "script": "import sys\ns = sys.stdin.read().strip()\nprint(s[::-1])"
    }
]


BANGla_PROMPT_TEMPLATE: str = (
    "নীচের নির্দেশনা অনুযায়ী একটি Python স্ক্রিপ্ট লিখুন। "
    "স্ক্রিপ্টটি stdin থেকে পড়বে এবং stdout-এ ফলাফল প্রিন্ট করবে। "
    "শুধুমাত্র প্রয়োজনীয় কোড লিখুন।\n\n"
    "উদাহরণ:\n"
    "নির্দেশনা: দুটি পূর্ণসংখ্যার যোগফল নির্ণয় করো\n"
    "স্ক্রিপ্ট:\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n\n"
    "নির্দেশনা: একটি সংখ্যার বর্গ নির্ণয় করো\n"
    "স্ক্রিপ্ট:\nimport sys\nn = int(sys.stdin.read())\nprint(n * n)\n\n"
    "নির্দেশনা:\n{instruction}\n\n"
    "স্ক্রিপ্ট:\n"
)


def build_prompt(instruction: str) -> str:
    cleaned_instruction: str = (instruction or "").strip()
    return BANGla_PROMPT_TEMPLATE.format(instruction=cleaned_instruction)


def build_few_shot_prompt(instruction: str, num_examples: int = 3) -> str:
    """Build prompt with multiple few-shot examples"""
    import random
    examples = random.sample(FEW_SHOT_EXAMPLES, min(num_examples, len(FEW_SHOT_EXAMPLES)))
    
    prompt = (
        "নীচের নির্দেশনা অনুযায়ী একটি Python স্ক্রিপ্ট লিখুন। "
        "স্ক্রিপ্টটি stdin থেকে পড়বে এবং stdout-এ ফলাফল প্রিন্ট করবে।\n\n"
    )
    
    for ex in examples:
        prompt += f"নির্দেশনা: {ex['instruction']}\n"
        prompt += f"স্ক্রিপ্ট:\n{ex['script']}\n\n"
    
    prompt += f"নির্দেশনা:\n{instruction}\n\n"
    prompt += "স্ক্রিপ্ট:\n"
    return prompt


@dataclass(frozen=True)
class StopConfig:
    eos_token_texts: tuple[str, ...] = ("```",)
    max_new_tokens: int = 512


DEFAULT_STOP_CONFIG = StopConfig()