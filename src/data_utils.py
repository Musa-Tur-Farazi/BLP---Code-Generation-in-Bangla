from __future__ import annotations

import json
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BlpItem:
    id: int
    instruction: str
    response: Optional[str] = None
    test_list: Optional[str] = None


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
        test_list=obj.get("test_list"),
    )


def load_items(path: str) -> List[BlpItem]:
    raw = read_json(path)
    items = [parse_item(x) for x in raw]
    return items


def parse_test_list(test_list_value: Optional[str]) -> List[str]:
    if not test_list_value:
        return []
    candidate = test_list_value
    # Some datasets double-quote the list string; try up to two safe eval passes
    for _ in range(2):
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            break
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            candidate = parsed
            continue
        break
    return []