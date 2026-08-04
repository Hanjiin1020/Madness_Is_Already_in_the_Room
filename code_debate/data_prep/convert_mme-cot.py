#!/usr/bin/env python3
"""Convert MME-CoT to the debate runner schema."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from _convert_common import LETTERS, choice_columns, load_ids, normalize_image, read_sources, save

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "MME-CoT_converted.parquet"
DEFAULT_IDS = ROOT.parent / "data_results_analysis" / "clean_question_ids_mmecot.json"

def convert(source: pd.DataFrame, ids: set[str] | None) -> pd.DataFrame:
    records = []
    for _, row in source.iterrows():
        question_id = str(row["index"])
        if ids is not None and question_id not in ids:
            continue
        choices = choice_columns(row)
        record = {
            "index": question_id,
            "question": re.sub(r"<image[^>]*>", "", str(row["question"])).strip(),
            "image": normalize_image(row["image"], row["_source_dir"]),
            "answer": str(row["answer"]).strip(),
            "answer_type": "mcq" if choices else "open",
            "category": str(row.get("category", "")),
            "category_detail": str(row.get("subcategory", "")),
            "question_type": str(row.get("question_type", "")),
        }
        for letter in LETTERS:
            record[letter] = choices.get(letter, "nan")
        records.append(record)
    frame = pd.DataFrame(records)
    if ids is not None and set(frame["index"].astype(str)) != ids:
        missing = sorted(ids - set(frame["index"].astype(str)))
        raise ValueError(f"Requested MME-CoT IDs were not found: {missing[:10]}")
    return frame

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MME-CoT for the debate runner")
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS if DEFAULT_IDS.exists() else None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    frame = convert(read_sources(args.src), load_ids(args.ids))
    save(frame, args.out)
    print(f"Converted {len(frame):,} MME-CoT rows -> {args.out}")

if __name__ == "__main__":
    main()
