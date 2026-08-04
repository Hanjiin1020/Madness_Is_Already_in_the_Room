"""Shared dataset conversion helpers."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pandas as pd

LETTERS = "ABCDEFGHIJKL"
MISSING = {"", "nan", "None", "null", "<NA>"}

def read_sources(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix == ".jsonl":
            frame = pd.read_json(path, lines=True)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported source format: {path}")
        frame = frame.copy()
        frame["_source_dir"] = str(path.resolve().parent)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one source file is required")
    return pd.concat(frames, ignore_index=True, sort=False)

def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() in MISSING

def unwrap(value):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        if value.get("path"):
            return value["path"]
    if not isinstance(value, (str, bytes, bytearray, dict)) and hasattr(value, "__len__"):
        if len(value) == 1:
            return value[0]
    return value

def normalize_image(value, source_dir: str | Path):
    value = unwrap(value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    if value.startswith(("iVBOR", "/9j/", "R0lGOD")) or len(value) > 1024:
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(source_dir) / candidate
    if candidate.is_file():
        return candidate.read_bytes()
    return value

def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ID file must contain a JSON list: {path}")
    return {str(value) for value in payload}

def choice_columns(row) -> dict[str, str]:
    choices = {}
    for letter in LETTERS:
        if letter not in row.index or is_missing(row[letter]):
            continue
        choices[letter] = str(row[letter]).strip()
    return choices

def validate(frame: pd.DataFrame) -> None:
    required = {"index", "question", "image", "answer", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing output columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Conversion produced no rows")
    if not frame["index"].astype(str).is_unique:
        raise ValueError("Output question IDs are not unique")
    if frame["question"].astype(str).str.strip().eq("").any():
        raise ValueError("Output contains an empty question")
    if frame["image"].map(is_missing).any():
        raise ValueError("Output contains a missing image")
    for row in frame.itertuples(index=False):
        values = row._asdict()
        choices = [letter for letter in LETTERS
                   if letter in values and not is_missing(values[letter])]
        if choices and choices != list(LETTERS[:len(choices)]):
            raise ValueError(f"Non-contiguous choices for question {values['index']}")
        answer = str(values["answer"]).strip()
        if choices and answer not in choices:
            raise ValueError(f"Answer {answer!r} is not a choice for question {values['index']}")

def verify_images(frame: pd.DataFrame, samples: int = 10) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    positions = sorted(set(int(i) for i in pd.Series(range(len(frame))).sample(
        n=min(samples, len(frame)), random_state=0)))
    for position in positions:
        value = unwrap(frame.iloc[position].image)
        if isinstance(value, str):
            value = base64.b64decode(value)
        Image.open(io.BytesIO(value)).verify()

def save(frame: pd.DataFrame, path: Path) -> None:
    validate(frame)
    verify_images(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    if len(restored) != len(frame) or not restored["index"].astype(str).is_unique:
        raise ValueError("Parquet round-trip validation failed")
