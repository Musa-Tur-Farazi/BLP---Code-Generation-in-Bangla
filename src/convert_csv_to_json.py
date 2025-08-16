from __future__ import annotations

import argparse
import csv
import json
from typing import Optional, Dict, Any, List


def try_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        # Fallback: attempt safe float->int for cases like '231.0'
        try:
            f = float(value)
            if f.is_integer():
                return int(f)
        except Exception:
            pass
        raise ValueError(f"Invalid integer id: {value!r}")


def convert_row(row: Dict[str, str], id_col: str, instruction_col: str, response_col: Optional[str], test_cases_col: Optional[str]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    obj["id"] = try_int((row.get(id_col) or "").strip())
    obj["instruction"] = (row.get(instruction_col) or "").strip()

    if response_col:
        resp = row.get(response_col)
        obj["response"] = None if resp is None or resp == "" else str(resp)
    else:
        obj["response"] = None

    if test_cases_col:
        tc = row.get(test_cases_col)
        obj["test_cases"] = None if tc is None or tc == "" else str(tc)
    else:
        obj["test_cases"] = None

    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert BLP CSV to JSON format")
    p.add_argument("--input_csv", required=True, type=str)
    p.add_argument("--output_json", required=True, type=str)
    p.add_argument("--id_col", type=str, default="id")
    p.add_argument("--instruction_col", type=str, default="instruction")
    p.add_argument("--response_col", type=str, default="response")
    p.add_argument("--test_cases_col", type=str, default="test_cases")
    p.add_argument("--encoding", type=str, default="utf-8")
    p.add_argument("--delimiter", type=str, default=",")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.input_csv, "r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=args.delimiter)
        headers = reader.fieldnames or []
        required = [args.id_col, args.instruction_col]
        for col in required:
            if col not in headers:
                raise ValueError(f"Missing required column: {col!r}. Found: {headers}")

        for row in reader:
            obj = convert_row(
                row=row,
                id_col=args.id_col,
                instruction_col=args.instruction_col,
                response_col=args.response_col if args.response_col in headers else None,
                test_cases_col=args.test_cases_col if args.test_cases_col in headers else None,
            )
            rows.append(obj)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} examples to {args.output_json}")


if __name__ == "__main__":
    main()