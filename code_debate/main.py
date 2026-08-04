

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mllm_backend import (  # noqa: E402
    MODELS, build_question, image_b64_from_row, infer_question_type,
    parse_final, VLLMClient, log,
)

from src.prompt_builder import build_vanilla_prompt  # noqa: E402

def parse_agent_spec(spec: str) -> dict:
    if spec not in MODELS:
        raise ValueError(f"모르는 model_key: {spec!r} (가능: {list(MODELS)})")
    return {"id": spec, "model_key": spec}

def make_messages(prompt_text: str, image_b64: str | None) -> list[dict]:
    if image_b64 is None:
        return [{"role": "user", "content": prompt_text}]
    mime = "image/png" if image_b64.startswith("iVBOR") else "image/jpeg"
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ]}]

def prev_block_selfother(round_out: dict, self_id: str) -> str:
    self_resp = round_out[self_id]["response"]
    others = [f"--- Agent: {aid} ---\n{d['response']}"
              for aid, d in round_out.items() if aid != self_id]
    return (f"[YOUR PREVIOUS RESPONSE]\n{self_resp}\n\n"
            f"[OTHER AGENTS' PREVIOUS RESPONSES]\n" + "\n\n".join(others))

def prev_block_self(round_out: dict, self_id: str) -> str:

    return f"[YOUR PREVIOUS RESPONSE]\n{round_out[self_id]['response']}"

def build_prompt(agent: dict, q_text: str, prev_block: str | None,
                 round_: int, qtype: str) -> str:
    return build_vanilla_prompt(q_text, prev_block=prev_block, round_=round_, qtype=qtype)

async def debate_one(row, sid_col, img_col, q_col, agents, clients, rounds,
                     answer_mode, sampling, mode="debate"):
    sid = str(row[sid_col])
    q_text, opts = build_question(row[q_col], row)
    image_b64 = image_b64_from_row(row[img_col])
    qtype = infer_question_type(row, opts, answer_mode)
    valid = set(opts) if opts else set()
    if opts:
        q_text = q_text + "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())

    
    prev_fn = prev_block_self if mode == "self_reflect" else prev_block_selfother
    transcript, prev_round = [], None
    for r in range(rounds):
        async def one_agent(ag):
            prev_block = None if r == 0 else prev_fn(prev_round, ag["id"])
            prompt = build_prompt(ag, q_text, prev_block, r, qtype)
            msgs = make_messages(prompt, image_b64)
            cli = clients[ag["model_key"]]
            mt = min(sampling["max_tokens"], MODELS[ag["model_key"]][4])  
            
            mint = sampling["min_tokens"] if ag["model_key"] in sampling["min_tokens_models"] else None
            try:
                content, finish, toks = await cli.achat(
                    msgs, mt, sampling["temperature"],
                    top_p=sampling["top_p"], seed=sampling["seed"], min_tokens=mint)
            except Exception as e:
                
                
                content, finish, toks = f"[NO_RESPONSE: {type(e).__name__}]", "error", 0
            return ag["id"], content, finish, toks
        results = await asyncio.gather(*[one_agent(a) for a in agents])
        round_out = {aid: {"response": c, "finish": f, "tokens": t}
                     for aid, c, f, t in results}
        transcript.append({"round": r, "outputs": round_out})
        prev_round = round_out

    
    final_answers = {}
    for ag in agents:
        content = prev_round[ag["id"]]["response"]
        if content.startswith("[NO_RESPONSE"):
            continue  
        atext, letter, _ = parse_final(content, valid)
        final_answers[ag["id"]] = (letter if (valid and letter) else atext)
    return {"sample_id": sid, "qtype": qtype, "gt": row.get("gt", row.get("answer")),
            "category": row.get("category"), "final_answers": final_answers,
            "transcript": transcript, "valid": sorted(valid)}

async def run(args):
    agents = [parse_agent_spec(s) for s in args.agents]
    used_models = sorted({a["model_key"] for a in agents})
    
    if args.mode == "self_reflect" and len(agents) != 1:
        log.error(f"[중단] mode={args.mode} 는 single 모델 전용 — --agents 는 1개만 (지금 {len(agents)}개)")
        return
    log.info(f"mode={args.mode} | 에이전트 {len(agents)}: {[a['id'] for a in agents]} | 모델 {used_models}")

    
    clients = {}
    for mk in used_models:
        name, url, think_off, default_conc, cap = MODELS[mk]
        clients[mk] = VLLMClient(name, url, think_off, args.max_concurrency or default_conc)
        if not await clients[mk].ping():
            log.error(f"[{mk}] 서버 다운 @ {url} — 중단")
            return
    
    mt_models = set(args.min_tokens_models or []) if args.min_tokens else set()
    if args.min_tokens and not mt_models:
        mt_models = set(used_models)          
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                "max_tokens": args.max_tokens,
                "min_tokens": args.min_tokens, "min_tokens_models": mt_models}
    if args.min_tokens:
        log.info(f"min_tokens={args.min_tokens} 적용 대상: {sorted(mt_models)}")

    df = pd.read_parquet(args.parquet)
    if args.n_samples:
        df = df.head(args.n_samples)

    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debate.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done.add(str(json.loads(line)["sample_id"]))
                except Exception: pass
    todo = [row for _, row in df.iterrows() if str(row[args.sid_col]) not in done]
    log.info(f"todo={len(todo)} (skip {len(done)}) rounds={args.rounds} → {out_path}")

    (out_dir / "run_meta.json").write_text(json.dumps({
        "parquet": args.parquet, "agents": [a["id"] for a in agents], "rounds": args.rounds,
        "mode": args.mode,
        "aggregation": "post-hoc (토론 후 분석 단계에서 집계)", "answer_mode": args.answer_mode,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens, "min_tokens_models": sorted(mt_models),
        "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed,
        "prompt": "src/prompt_builder (우리 CoT 기준, SG 없음, self/other 토론)",
    }, ensure_ascii=False, indent=2))

    write_lock = asyncio.Lock()
    out_f = open(out_path, "a", encoding="utf-8")
    rec_sem = asyncio.Semaphore(args.max_records_in_flight) if args.max_records_in_flight else None
    n = [0]

    async def process(row):
        async def _go():
            res = await debate_one(row, args.sid_col, args.img_col, args.q_col,
                                   agents, clients, args.rounds, args.answer_mode, sampling,
                                   mode=args.mode)
            async with write_lock:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n[0] += 1
                if n[0] % 10 == 0:
                    log.info(f"진행 {n[0]}/{len(todo)}")
        try:
            if rec_sem:
                async with rec_sem: await _go()
            else:
                await _go()
        except Exception as e:
            log.warning(f"[skip] sample={row[args.sid_col]}: {type(e).__name__}: {str(e)[:150]}")

    await asyncio.gather(*[process(r) for r in todo])
    out_f.close()
    log.info(f"완료 +{n[0]} → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="vLLM 서빙 기반 Multi-Agent Debate 실행")
    ap.add_argument("--parquet", required=True,
                    help="입력 parquet (예: data/MMStar_converted.parquet)")
    ap.add_argument("--agents", nargs="+", required=True,
                    help='model_key 목록. 예: gemma4 internvl3_5 qwen35')
    ap.add_argument("--rounds", type=int, default=2, help="총 라운드 수(0=독립, 이후 토론). 기본 2")
    ap.add_argument("--mode", choices=["debate", "self_reflect"], default="debate",
                    help="debate=토론 / self_reflect=single, 이전라운드 자기응답만(others 삭제)")
    ap.add_argument("--answer-mode", choices=["auto", "mcq", "open", "binary"], default="auto")
    ap.add_argument("--sid-col", default="index")
    ap.add_argument("--img-col", default="image")
    ap.add_argument("--q-col", default="question")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="vLLM min_tokens(이 토큰 수 전까지 EOS 금지). 0=미적용")
    ap.add_argument("--min-tokens-models", nargs="*", default=None,
                    help="min_tokens 를 적용할 model_key 목록(기본: 전 모델)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-concurrency", type=int, default=0, help="0=모델별 기본값")
    ap.add_argument("--max-records-in-flight", type=int, default=48,
                    help="동시 처리 샘플 수 상한(점진 flush). 0=무제한")
    ap.add_argument("--n-samples", type=int, default=0, help="상위 N개만(점검용)")
    ap.add_argument("--out-dir", default="mad_outputs")
    ap.add_argument("--run-id", default="debate_v1")
    args = ap.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
