from __future__ import annotations

import argparse
import ast
import builtins
import io
import sys
from contextlib import redirect_stdout
from typing import Dict, List

from .data_utils import load_items, parse_test_list


SAFE_MODULES = {
    "math",
    "re",
    "collections",
    "heapq",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "bisect",
    "string",
    "random",
}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"import of module '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__import__": safe_import,
    "abs": builtins.abs,
    "min": builtins.min,
    "max": builtins.max,
    "range": builtins.range,
    "len": builtins.len,
    "sum": builtins.sum,
    "print": builtins.print,
    "enumerate": builtins.enumerate,
    "zip": builtins.zip,
    "map": builtins.map,
    "filter": builtins.filter,
    "all": builtins.all,
    "any": builtins.any,
    "sorted": builtins.sorted,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "pow": builtins.pow,
    "ord": builtins.ord,
    "chr": builtins.chr,
    "list": builtins.list,
    "dict": builtins.dict,
    "set": builtins.set,
    "tuple": builtins.tuple,
}


class RestrictedDict(dict):
    def __getitem__(self, key):
        if key in ("open", "exec", "eval", "compile"):
            raise KeyError("forbidden")
        return super().__getitem__(key)


def run_function_and_asserts(func_src: str, asserts: List[str]) -> Dict[str, int]:
    globals_dict: Dict[str, object] = RestrictedDict({"__builtins__": SAFE_BUILTINS})
    locals_dict: Dict[str, object] = {}

    # Define the function in the restricted namespace
    try:
        compiled = compile(func_src, filename="<pred>", mode="exec")
        exec(compiled, globals_dict, locals_dict)
    except Exception:
        return {"passed": 0, "failed": len(asserts)}

    # Run asserts
    passed = 0
    for stmt in asserts:
        s = stmt.strip()
        if not s.startswith("assert "):
            s = "assert " + s
        try:
            compiled_assert = compile(s, filename="<assert>", mode="exec")
            exec(compiled_assert, {**globals_dict, **locals_dict}, {})
            passed += 1
        except Exception:
            pass

    return {"passed": passed, "failed": len(asserts) - passed}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated functions against dev asserts")
    p.add_argument("--data_json", required=True, type=str)
    p.add_argument("--pred_json", required=True, type=str)
    return p.parse_args()


def main() -> None:
    import json

    args = parse_args()
    data_items = {it.id: it for it in load_items(args.data_json)}

    with open(args.pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    total_passed, total_failed = 0, 0
    for obj in preds:
        pid = int(obj["id"])
        func_src = str(obj["response"]) if obj.get("response") is not None else ""
        asserts = parse_test_list(data_items.get(pid).test_list if pid in data_items else None)
        if not asserts:
            continue
        res = run_function_and_asserts(func_src, asserts)
        total_passed += res["passed"]
        total_failed += res["failed"]

    print(f"Passed: {total_passed} | Failed: {total_failed}")


if __name__ == "__main__":
    main()