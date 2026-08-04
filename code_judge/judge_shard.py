#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("sej", str(HERE / "src" / "score_equiv_judge.py"))
sej = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sej)

_jf = importlib.util.spec_from_file_location("jf", str(HERE / "judge_freeform.py"))
jf = importlib.util.module_from_spec(_jf)
_jf.loader.exec_module(jf)

def shard_of(key: str, n: int) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % n

async def main() -> None:
    ap = argparse.ArgumentParser(description="주관식 등가판정 — 샤드 워커")
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--out-cache", required=True, type=Path, help="이 샤드의 판정 결과")
    ap.add_argument("--cache", type=Path, default=HERE / "equiv_cache.json",
                    help="기존 공유 캐시 (읽기 전용 — 이미 판정된 쌍 건너뛰기)")
    ap.add_argument("--debate-code", type=Path, default=HERE.parent / "code_debate")
    ap.add_argument("--judge-base", required=True)
    ap.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    ap.add_argument("--max-conc", type=int, default=64)
    args = ap.parse_args()

    parse_final = jf.load_parse_final(args.debate_code)
    base_cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    pairs = jf.collect(args.runs_dir, parse_final)

    results, todo = {}, []
    for gold, pred in sorted(pairs):
        key = f"{gold}\t{pred}"
        if key in base_cache:
            continue
        if shard_of(key, args.nshards) != args.shard:
            continue
        if sej.normalize_answer(pred) in sej._gold_answer_variants(gold):
            results[key] = True            
        else:
            todo.append((gold, pred, key))
    print(f"[shard {args.shard}/{args.nshards}] exact {len(results):,} | judge 대상 {len(todo):,}",
          flush=True)

    scorer = sej.EquivScorer(args.judge_base, args.judge_model, max_conc=args.max_conc)
    done = [0]

    async def one(g, p, k):
        try:
            results[k] = await scorer.equiv(g, p)
        except Exception:
            results[k] = False             
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  [shard {args.shard}] 판정 {done[0]:,}/{len(todo):,}", flush=True)

    if todo:
        await asyncio.gather(*[one(g, p, k) for g, p, k in todo])
    args.out_cache.write_text(json.dumps(results, ensure_ascii=False))
    yes = sum(1 for v in results.values() if v)
    print(f"[shard {args.shard}] 완료: yes {yes:,} / no {len(results) - yes:,} → {args.out_cache}",
          flush=True)

if __name__ == "__main__":
    asyncio.run(main())
