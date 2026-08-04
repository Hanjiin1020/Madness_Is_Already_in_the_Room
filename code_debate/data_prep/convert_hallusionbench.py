#!/usr/bin/env python3
"""Convert HallusionBench to the debate runner schema."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from _convert_common import is_missing, load_ids, normalize_image, read_sources, save

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "HallusionBench_converted.parquet"
DEFAULT_IDS = ROOT.parent / "data_results_analysis" / "clean_question_ids_hallusionbench.json"

def first_present(row: pd.Series, names: list[str]):
    for name in names:
        if name in row.index and not is_missing(row[name]):
            return row[name]
    return None

def convert(source: pd.DataFrame, ids: set[str] | None) -> pd.DataFrame:
    records = []
    for _, row in source.iterrows():
        question_id = str(first_present(row, ["index", "id", "question_id"]))
        if ids is not None and question_id not in ids:
            continue
        answer = str(first_present(row, ["answer", "gt_answer_details"])).strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError(f"Invalid binary answer for {question_id}: {answer!r}")
        image_value = first_present(row, ["image", "image_path", "filename"])
        visual_label = str(first_present(row, ["vd_vs", "visual_label"]) or "").upper()
        records.append({
            "index": question_id,
            "question": re.sub(r"<image[^>]*>", "", str(row["question"])).strip(),
            "image": normalize_image(image_value, row["_source_dir"]),
            "answer": answer.title(),
            "answer_type": "binary",
            "category": visual_label,
            "category_detail": str(first_present(row, ["category", "subcategory"]) or ""),
            "image_id": str(first_present(row, ["image_id", "figure_id"]) or ""),
        })
    frame = pd.DataFrame(records)
    if ids is not None and set(frame["index"].astype(str)) != ids:
        missing = sorted(ids - set(frame["index"].astype(str)))
        raise ValueError(f"Requested HallusionBench IDs were not found: {missing[:10]}")
    return frame

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HallusionBench for the debate runner")
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--ids", type=Path,
                        default=DEFAULT_IDS if DEFAULT_IDS.exists() else None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    frame = convert(read_sources(args.src), load_ids(args.ids))
    save(frame, args.out)
    print(f"Converted {len(frame):,} HallusionBench rows -> {args.out}")

if __name__ == "__main__":
    main()
