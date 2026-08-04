#!/usr/bin/env python3
"""Convert MMStar to the debate runner schema."""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd

from _convert_common import LETTERS, is_missing, load_ids, normalize_image, read_sources, save

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "MMStar_converted.parquet"
DEFAULT_IDS = ROOT.parent / "data_results_analysis" / "clean_question_ids_mmstar.json"

def find_source() -> Path:
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        hf_home = os.environ.get("HF_HOME")
        cache = str(Path(hf_home) / "hub") if hf_home else str(
            Path.home() / ".cache" / "huggingface" / "hub")
    pattern = str(Path(cache) / "datasets--Lin-Chen--MMStar" /
                  "snapshots" / "*" / "mmstar.parquet")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "MMStar source not found. Download Lin-Chen/MMStar or pass --src.")
    return Path(matches[0])

def split_block(question: str) -> tuple[str, str | None, str]:
    match = re.search(r"\n\s*Options:\s*", question)
    if match:
        return question[:match.start()], question[match.end():], "colon"
    match = re.search(r"\n\s*Choices:\s*\n?", question)
    if match:
        return question[:match.start()], question[match.end():], "paren"
    match = re.search(r"(?m)^\(A\)\s", question)
    if match:
        return question[:match.start()], question[match.start():], "paren"
    return question, None, "none"

def parse_colon(block: str) -> dict[str, str]:
    choices = {}
    start = block.find("A:")
    if start < 0:
        return choices
    current, offset = "A", start + 2
    for letter in LETTERS[1:]:
        match = re.search(r"[,\n]\s*" + letter + r":\s*", block[offset:])
        if not match:
            choices[current] = block[offset:].strip().strip(",").strip()
            break
        choices[current] = block[offset:offset + match.start()].strip().strip(",").strip()
        current, offset = letter, offset + match.end()
    else:
        choices[current] = block[offset:].strip()
    return choices

def parse_parenthesized(block: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    current = None
    for line in block.splitlines():
        match = re.match(r"^\s*\(([A-L])\)\s*(.*)$", line)
        if match:
            current = match.group(1)
            choices[current] = match.group(2).strip()
        elif current and line.strip():
            choices[current] += " " + line.strip()
    return choices

def normalize_choices(choices: dict[str, str], question_id: str) -> dict[str, str]:
    keys = sorted(choices)
    if "".join(keys) != LETTERS[:len(keys)]:
        raise ValueError(f"Non-contiguous choices for question {question_id}: {keys}")
    present = [position for position, key in enumerate(keys)
               if not is_missing(choices[key])]
    if not present:
        raise ValueError(f"No choices found for question {question_id}")
    for position, key in enumerate(keys[:max(present) + 1]):
        if is_missing(choices[key]):
            choices[key] = "none"
    return choices

def convert(source: pd.DataFrame, ids: set[str] | None) -> pd.DataFrame:
    records = []
    for _, row in source.iterrows():
        question_id = str(row["index"])
        if ids is not None and question_id not in ids:
            continue
        body, block, style = split_block(str(row["question"]))
        if block is None:
            choices = {letter: str(row[letter]).strip() for letter in LETTERS
                       if letter in row.index and not is_missing(row[letter])}
            style = str(row.get("opt_style", "columns"))
        else:
            choices = parse_colon(block) if style == "colon" else parse_parenthesized(block)
        choices = normalize_choices(choices, question_id)
        real = {key: value for key, value in choices.items() if not is_missing(value)}
        answer = str(row["answer"]).strip().upper()
        if len(real) < 2 or answer not in real:
            raise ValueError(f"Invalid choices or answer for question {question_id}")

        record = {
            "index": question_id,
            "question": body.strip(),
            "image": normalize_image(row["image"], row["_source_dir"]),
            "answer": answer,
            "answer_type": "mcq",
            "category": str(row.get("category", "")),
            "l2_category": str(row.get("l2_category", "")),
            "meta_info": str(row.get("meta_info", "")),
            "n_options": len(real),
            "opt_style": style,
        }
        for letter in LETTERS:
            record[letter] = choices.get(letter, "nan")
        records.append(record)

    frame = pd.DataFrame(records)
    if ids is not None and set(frame["index"].astype(str)) != ids:
        missing = sorted(ids - set(frame["index"].astype(str)))
        raise ValueError(f"Requested MMStar IDs were not found: {missing[:10]}")
    return frame

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MMStar for the debate runner")
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--ids", type=Path,
                        default=DEFAULT_IDS if DEFAULT_IDS.exists() else None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = args.src or find_source()
    frame = convert(read_sources([source]), load_ids(args.ids))
    save(frame, args.out)
    print(f"Converted {len(frame):,} MMStar rows -> {args.out}")

if __name__ == "__main__":
    main()
