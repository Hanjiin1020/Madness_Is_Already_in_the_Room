#!/usr/bin/env python3
"""Convert single-image MMMU questions to the debate runner schema."""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pandas as pd

from _convert_common import LETTERS, is_missing, load_ids, normalize_image, read_sources, save

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "MMMU_converted.parquet"
DEFAULT_IDS = ROOT.parent / "data_results_analysis" / "clean_question_ids_mmmu.json"

def first_value(*values, default=""):
    for value in values:
        if not is_missing(value):
            return value
    return default

def infer_subject(question_id: str) -> str:
    parts = question_id.split("_")
    return "_".join(parts[1:-1]) if len(parts) > 2 else ""

def parse_options(row: pd.Series) -> dict[str, str]:
    choices = {}
    for letter in LETTERS:
        if letter in row.index and not is_missing(row[letter]):
            choices[letter] = str(row[letter]).strip()
    if choices:
        return choices
    value = row.get("options")
    if is_missing(value):
        return {}
    parsed = ast.literal_eval(value) if isinstance(value, str) else list(value)
    return {letter: str(option).strip() for letter, option in zip(LETTERS, parsed)}

def select_image(row: pd.Series):
    if "image" in row.index and not is_missing(row["image"]):
        return row["image"]
    candidates = [row[name] for name in row.index
                  if re.fullmatch(r"image_\d+", str(name)) and not is_missing(row[name])]
    return candidates[0] if len(candidates) == 1 else None

def prepare_rows(source: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source["_question_id"] = source.apply(
        lambda row: str(row["id"] if "id" in row.index else row["index"]), axis=1)
    source["_subject"] = source.apply(
        lambda row: str(first_value(
            row.get("subject"), infer_subject(row["_question_id"]))), axis=1)
    source["_image"] = source.apply(select_image, axis=1)
    return source[~source["_image"].map(is_missing)].copy()

def sample_subjects(source: pd.DataFrame, per_subject: int, seed: int) -> pd.DataFrame:
    if per_subject <= 0:
        return source
    counts = source.groupby("_subject").size()
    too_small = counts[counts < per_subject]
    if not too_small.empty:
        raise ValueError(f"Subjects below --per-subject={per_subject}: {too_small.to_dict()}")
    return (source.groupby("_subject", group_keys=False)
            .sample(n=per_subject, random_state=seed))

def convert(source: pd.DataFrame, ids: set[str] | None,
            per_subject: int, seed: int) -> pd.DataFrame:
    source = prepare_rows(source)
    if ids is not None:
        source = source[source["_question_id"].isin(ids)].copy()
    else:
        source = sample_subjects(source, per_subject, seed)

    records = []
    for _, row in source.iterrows():
        choices = parse_options(row)
        answer = str(row["answer"]).strip().upper()
        record = {
            "index": row["_question_id"],
            "question": re.sub(r"<image[^>]*>", "", str(row["question"])).strip(),
            "image": normalize_image(row["_image"], row["_source_dir"]),
            "answer": answer,
            "answer_type": "mcq",
            "category": str(first_value(row.get("category"), row["_subject"])),
            "category_detail": str(first_value(row.get("subfield"))),
            "subject": row["_subject"],
            "split": str(first_value(
                row.get("split"), row["_question_id"].split("_", 1)[0])),
        }
        for letter in LETTERS:
            record[letter] = choices.get(letter, "nan")
        records.append(record)
    frame = pd.DataFrame(records).sort_values("index").reset_index(drop=True)
    if ids is not None and set(frame["index"].astype(str)) != ids:
        missing = sorted(ids - set(frame["index"].astype(str)))
        raise ValueError(f"Requested MMMU IDs were not found: {missing[:10]}")
    return frame

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MMMU for the debate runner")
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--ids", type=Path,
                        default=DEFAULT_IDS if DEFAULT_IDS.exists() else None)
    parser.add_argument("--per-subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    frame = convert(read_sources(args.src), load_ids(args.ids), args.per_subject, args.seed)
    save(frame, args.out)
    print(f"Converted {len(frame):,} MMMU rows -> {args.out}")

if __name__ == "__main__":
    main()
