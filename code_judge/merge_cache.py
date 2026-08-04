#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description="판정 캐시 병합")
    ap.add_argument("shards", nargs="+", type=Path, help="샤드 결과 JSON 목록")
    ap.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / "equiv_cache.json")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    before = len(cache)
    if cache and not args.no_backup:
        bak = args.cache.with_suffix(".json.bak")
        shutil.copyfile(args.cache, bak)
        print(f"백업 → {bak}")

    added = 0
    for f in args.shards:
        if not f.exists():
            print(f"  [없음] {f}")
            continue
        d = json.loads(f.read_text())
        for k, v in d.items():
            if k not in cache:
                added += 1
            cache[k] = v
        print(f"  병합 {f} ({len(d):,}쌍)")

    args.cache.write_text(json.dumps(cache, ensure_ascii=False))
    yes = sum(1 for v in cache.values() if v)
    print(f"\n캐시 {before:,} → {len(cache):,} (신규 {added:,}) | yes {yes:,} / no {len(cache)-yes:,}")
    print(f"저장 → {args.cache}")

if __name__ == "__main__":
    main()
