from __future__ import annotations

import json
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class BlpItem:
    id: int
    instruction: str
    response: Optional[str] = None
    test_cases: Optional[str] = None  # Changed from test_list


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a list of objects")
    return data


def parse_item(obj: Dict[str, Any]) -> BlpItem:
    return BlpItem(
        id=int(obj.get("id")),
        instruction=str(obj.get("instruction", "")),
        response=obj.get("response"),
        test_cases=obj.get("test_cases"),  # Changed from test_list
    )


def load_items(path: str) -> List[BlpItem]:
    raw = read_json(path)
    items = [parse_item(x) for x in raw]
    return items


def load_dataset(path: str) -> List[BlpItem]:
    """Load dataset from JSON file - alias for load_items for compatibility"""
    return load_items(path)


def parse_test_cases(test_cases_value: Optional[str]) -> List[Tuple[str, str]]:
    """Parse test_cases as I/O pairs instead of assert statements"""
    if not test_cases_value:
        return []
    candidate = test_cases_value
    # Some datasets double-quote the list string; try up to two safe eval passes
    for _ in range(2):
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            break
        if isinstance(parsed, list):
            # Convert to list of (input, output) tuples
            io_pairs = []
            for pair in parsed:
                if isinstance(pair, list) and len(pair) == 2:
                    io_pairs.append((str(pair[0]), str(pair[1])))
            return io_pairs
        if isinstance(parsed, str):
            candidate = parsed
            continue
        break
    return []


def extract_keywords(text: str) -> List[str]:
    """Extract key programming concepts from text"""
    # Common programming keywords in Bangla/English
    keywords = [
        'array', 'list', 'string', 'number', 'count', 'sum', 'max', 'min',
        'sort', 'reverse', 'find', 'check', 'calculate', 'compute',
        'অ্যারে', 'তালিকা', 'স্ট্রিং', 'সংখ্যা', 'গণনা', 'যোগফল', 'সর্বোচ্চ', 'সর্বনিম্ন',
        'সাজানো', 'বিপরীত', 'খুঁজুন', 'পরীক্ষা', 'হিসাব', 'গণনা', 'গ.সা.গু', 'ল.সা.গু'
    ]
    
    found = []
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found.append(keyword)
    return found


def find_similar_examples(query_instruction: str, training_items: List[BlpItem], top_k: int = 3) -> List[BlpItem]:
    """Find similar examples based on keyword overlap"""
    query_keywords = set(extract_keywords(query_instruction))
    
    similarities = []
    for item in training_items:
        if not item.response:  # Skip items without responses
            continue
            
        item_keywords = set(extract_keywords(item.instruction))
        overlap = len(query_keywords.intersection(item_keywords))
        similarities.append((overlap, item))
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in similarities[:top_k]]


def build_rag_prompt(instruction: str, similar_examples: List[BlpItem]) -> str:
    """Build prompt with retrieved similar examples"""
    prompt = (
        "নীচের নির্দেশনা অনুযায়ী একটি Python স্ক্রিপ্ট লিখুন। "
        "অনুরূপ উদাহরণগুলি দেখুন এবং একই প্যাটার্ন অনুসরণ করুন।\n\n"
    )
    
    for i, example in enumerate(similar_examples, 1):
        prompt += f"উদাহরণ {i}:\n"
        prompt += f"নির্দেশনা: {example.instruction}\n"
        prompt += f"স্ক্রিপ্ট:\n{example.response}\n\n"
    
    prompt += f"নির্দেশনা:\n{instruction}\n\n"
    prompt += "স্ক্রিপ্ট:\n"
    return prompt