#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("sej", str(HERE / "src" / "score_equiv_judge.py"))
sej = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sej)

def load_parse_final(debate_code: Path):
    backend = debate_code / "src" / "mllm_backend.py"
    if not backend.exists():
        sys.exit(f"[중단] parse_final 을 찾을 수 없습니다: {backend}\n"
                 f"  --debate-code 로 토론 실행 코드 디렉토리를 지정하세요.")
    sys.path.insert(0, str(debate_code))
    spec = importlib.util.spec_from_file_location("mllm_backend", str(backend))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_final

def collect(runs_dir: Path, parse_final) -> set:
    pairs = set()
    files = sorted(runs_dir.glob("**/debate.jsonl"))
    if not files:
        sys.exit(f"[중단] debate.jsonl 을 찾을 수 없습니다: {runs_dir}")
    for p in files:
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("valid"):        
                    continue
                gt = str(rec.get("gt"))
                for rd in rec.get("transcript", []):
                    for _ag, o in rd.get("outputs", {}).items():
                        resp = (o or {}).get("response") or ""
                        if not resp.strip() or resp.startswith("[NO_RESPONSE"):
                            continue
                        atext, _letter, _m = parse_final(resp, set())
                        pred = (atext or "").strip()
                        if pred:
                            pairs.add((gt, pred))
    print(f"대상 파일 {len(files)}개 | open 유니크 쌍 {len(pairs):,}")
    return pairs

async def main() -> None:
    ap = argparse.ArgumentParser(description="주관식 답 등가판정 (judge LLM)")
    ap.add_argument("--runs-dir", required=True, type=Path,
                    help="토론 결과 디렉토리 (하위의 **/debate.jsonl 을 모두 읽음)")
    ap.add_argument("--cache", type=Path, default=HERE / "equiv_cache.json",
                    help="판정 캐시 JSON (증분 적재)")
    ap.add_argument("--debate-code", type=Path, default=HERE.parent / "code_debate",
                    help="토론 실행 코드 디렉토리 (parse_final 재사용)")
    ap.add_argument("--judge-base", default="http://localhost:8006/v1")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    ap.add_argument("--max-conc", type=int, default=64)
    args = ap.parse_args()

    parse_final = load_parse_final(args.debate_code)
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    pairs = collect(args.runs_dir, parse_final)
    print(f"캐시 보유 {len(cache):,}쌍")

    
    todo, n_exact = [], 0
    for gold, pred in sorted(pairs):
        key = f"{gold}\t{pred}"
        if key in cache:
            continue
        if sej.normalize_answer(pred) in sej._gold_answer_variants(gold):
            cache[key] = True
            n_exact += 1
        else:
            todo.append((gold, pred, key))
    print(f"exact 매칭 확정 {n_exact:,} | judge 호출 대상 {len(todo):,}")
    if not todo:
        args.cache.write_text(json.dumps(cache, ensure_ascii=False))
        print(f"judge 불필요 — 캐시 저장 완료 ({args.cache})")
        return

    
    scorer = sej.EquivScorer(args.judge_base, args.judge_model, max_conc=args.max_conc)
    done = [0]

    async def one(g, p, k):
        try:
            cache[k] = await scorer.equiv(g, p)
        except Exception:
            cache[k] = False          
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  판정 {done[0]:,}/{len(todo):,}", flush=True)

    await asyncio.gather(*[one(g, p, k) for g, p, k in todo])
    args.cache.write_text(json.dumps(cache, ensure_ascii=False))
    yes = sum(1 for _g, _p, k in todo if cache.get(k))
    print(f"\n판정 완료: yes {yes:,} / no {len(todo) - yes:,}  → 캐시 {len(cache):,}쌍 ({args.cache})")

if __name__ == "__main__":
    asyncio.run(main())
