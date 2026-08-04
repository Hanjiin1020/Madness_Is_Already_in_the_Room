

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mme_run")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

COT_SYSTEM = (
    "You are a vision-language agent answering a question about the given image. "
    "Look at the image carefully and reason step by step before answering."
)

ANSWER_FORMAT_MME = {
    "mcq":  "This is a multiple-choice question. Your final answer must be the single "
            "letter of the best option (e.g. 'A').",
    "open": "This is an open-ended question. Answer with a short, direct phrase, number, "
            "or expression.",
    
    "binary": "This is a yes/no question. Your final answer must be exactly 'Yes' or 'No'.",
}

OUTPUT_SCHEMA = (
    "First, think step by step and explain your reasoning based on the image. "
    "Then, on the last line, write exactly 'Answer: ' followed by your final answer."
)

MODELS = {
    "internvl3_5":   ("InternVL3_5-14B",       "http://localhost:8001/v1", False, 128, 4096),
    "gemma4":        ("gemma-4-12B-it",        "http://localhost:8003/v1", False, 128, 4096),
    "qwen35":        ("Qwen3.5-9B",            "http://localhost:8004/v1", True,  128, 16384),  
    "llama32vision": ("Llama-3.2-11B-Vision",  "http://localhost:8002/v1", False, 16,  4096),  
    
    
    "pixtral":       ("mistralai/Pixtral-12B-2409", "http://localhost:8011/v1", False, 16, 4096),
}

import os as _os
for _k in list(MODELS):
    _u = _os.environ.get(f"VLLM_URL_{_k}")
    if _u:
        _t = MODELS[_k]
        MODELS[_k] = (_t[0], _u, *_t[2:])

for _k in list(MODELS):
    _m = _os.environ.get(f"MAXTOK_{_k}")
    if _m:
        _t = MODELS[_k]
        MODELS[_k] = (*_t[:4], int(_m))
CHOICE_COLS = list("ABCDEFGHIJKL")
QUESTION_TYPES = {"mcq", "open", "binary"}

def image_b64_from_row(row_value) -> str | None:

    v = row_value
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        v = v[0]
    if isinstance(v, bytes):
        import base64
        return base64.b64encode(v).decode()
    return v if isinstance(v, str) and v else None

def build_question(question_text: str, row) -> tuple[str, dict]:
    q = re.sub(r"<image[^>]*>", "", str(question_text)).strip()
    opts = {}
    for c in CHOICE_COLS:
        try:
            val = str(row[c])
        except (KeyError, TypeError):
            continue
        if val not in ("nan", "None", ""):
            opts[c] = val
    return q, opts

def infer_question_type(row, opts: dict, answer_mode: str = "auto") -> str:
    if answer_mode in QUESTION_TYPES:
        return answer_mode
    if answer_mode != "auto":
        raise ValueError(f"Unknown answer mode: {answer_mode!r}")
    aliases = {
        "multiple_choice": "mcq",
        "multiple-choice": "mcq",
        "multiple choice": "mcq",
        "freeform": "open",
        "free-form": "open",
        "open-ended": "open",
        "open ended": "open",
        "yes_no": "binary",
        "yes/no": "binary",
    }
    for column in ("answer_type", "qtype", "question_type"):
        value = str(row.get(column, "")).strip().lower()
        value = aliases.get(value, value)
        if value in QUESTION_TYPES:
            return value
    visual_label = str(row.get("vd_vs", row.get("category", ""))).strip().upper()
    gold = str(row.get("answer", row.get("gt", ""))).strip().lower()
    if visual_label in {"VD", "VS"} and gold in {"yes", "no"}:
        return "binary"
    return "mcq" if opts else "open"

def build_messages(question: str, opts: dict, image_b64: str | None, answer_mode: str = "auto") -> list[dict]:
    qtype = answer_mode if answer_mode in QUESTION_TYPES else ("mcq" if opts else "open")
    txt = f"Question: {question}"
    if opts:
        txt += "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())
    
    
    txt += "\n\n" + ANSWER_FORMAT_MME[qtype] + "\n" + OUTPUT_SCHEMA
    if image_b64 is None:
        content = txt
    else:
        
        mime = "image/png" if image_b64.startswith("iVBOR") else "image/jpeg"
        content = [
            {"type": "text", "text": txt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]
    return [{"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": content}]

def _last_sentence(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    last_line = [ln for ln in text.splitlines() if ln.strip()][-1].strip()
    parts = re.split(r"(?<=[.?!])\s+", last_line)
    return (parts[-1].strip() if parts else last_line).strip("*").strip()

def parse_final(content: str, valid: set) -> tuple[str | None, str | None, str]:

    c = content or ""
    answer_text, method = None, None
    
    ms = list(re.finditer(r"(?im)^[\s*_#>\-]*Answer[\s*_]*:\s*(.+?)\s*$", c)) if answer_text is None else []
    if ms:
        cand = ms[-1].group(1).strip().strip("*").strip()
        if cand:
            answer_text, method = cand, "answer_line"
    
    if answer_text is None:
        pm = list(re.finditer(r"(?i)(?:final|correct)?\s*answer\s*(?:is|:)\s*\**\s*([^\n*.]{1,60})", c))
        if pm:
            cand = pm[-1].group(1).strip().strip("*").strip(" .:")
            if cand:
                answer_text, method = cand, "prose"
    
    if answer_text is None:
        answer_text = _last_sentence(c)
        method = "last_sentence"
    letter = None
    if valid:
        
        src = answer_text or ""
        for tok in re.findall(r"(?i)\b([A-L])\b", src):
            if tok.upper() in valid:
                letter = tok.upper(); break
        
        if letter is None:
            for pat in (
                r"(?i)answer\s*(?:is\s*:?|:)?\s*\**\s*\(?([A-L])\)?\b",  # "answer is C" / "answer: C" / "answer (C)"
                r"(?i)option\s*\**\s*\(?([A-L])\)?\b",                    # "option C"
                r"(?m)^\s*\**\s*\(?([A-L])[.)]",                          
            ):
                cands = [t.upper() for t in re.findall(pat, c) if t.upper() in valid]
                if cands:
                    letter = cands[-1]; break
    return answer_text, letter, method

class VLLMClient:
    def __init__(self, model_name, base_url, think_off, max_conc, max_retries=3, timeout=900.0):
        self.model_name = model_name
        self.think_off = think_off
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", max_retries=0, timeout=timeout)
        self.sem = asyncio.Semaphore(max_conc)
        self.max_retries = max_retries

    async def ping(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            log.error(f"server unreachable ({self.model_name}): {e}")
            return False

    async def achat(self, messages, max_tokens, temperature, top_p=None, seed=None,
                    min_tokens=None):

        kwargs = dict(model=self.model_name, messages=messages, max_tokens=max_tokens, temperature=temperature)
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None:
            kwargs["seed"] = seed
        extra = {}
        if self.think_off:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        if min_tokens:
            extra["min_tokens"] = int(min_tokens)
        if extra:
            kwargs["extra_body"] = extra
        last = None
        async with self.sem:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await self.client.chat.completions.create(**kwargs)
                    return (r.choices[0].message.content or "", r.choices[0].finish_reason,
                            r.usage.completion_tokens)
                except Exception as e:
                    last = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
        raise last

async def run_one_model(key, df, sid_col, img_col, q_col, out_path: Path,
                        max_tokens, temperature, max_conc, write_lock, dataset_name, seed, top_p,
                        answer_mode="auto"):
    name, url, think_off, default_conc, model_max_tok = MODELS[key]
    conc = max_conc if max_conc else default_conc
    max_tokens = min(max_tokens, model_max_tok)   
    client = VLLMClient(name, url, think_off, conc)
    if not await client.ping():
        log.error(f"[{key}] SKIP (server down @ {url})")
        return

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["sample_id"]))
                except Exception:
                    pass
    out_f = open(out_path, "a", encoding="utf-8")
    todo = [row for _, row in df.iterrows() if str(row[sid_col]) not in done]
    log.info(f"[{key}] model={name} conc={conc} think_off={think_off} "
             f"todo={len(todo)} (skip {len(done)})")

    n = [0]
    counter_lock = asyncio.Lock()

    async def gen(row):
        sid = str(row[sid_col])
        q, opts = build_question(row[q_col], row)
        b64 = image_b64_from_row(row[img_col])
        qtype = infer_question_type(row, opts, answer_mode)
        messages = build_messages(q, opts, b64, qtype)
        try:
            content, finish, toks = await client.achat(messages, max_tokens, temperature, top_p=top_p, seed=seed)
        except Exception as e:
            log.error(f"[{key}][{sid}] failed: {e}")
            return
        answer_text, letter, method = parse_final(content, set(opts))
        rec = {
            "dataset": dataset_name,        
            "sample_id": sid, "model_key": key, "model": name,
            "category": row.get("category"),
            "qtype": qtype,
            "gt": str(row.get("answer")),
            "answer_text": answer_text,     
            "letter": letter,               
            "parse_method": method,         
            "finish": finish, "tokens": toks, "response": content,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        async with write_lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
        async with counter_lock:
            n[0] += 1
            if n[0] % 50 == 0:
                log.info(f"[{key}] {n[0]}/{len(todo)}")

    await asyncio.gather(*[gen(r) for r in todo])
    out_f.close()
    log.info(f"[{key}] DONE +{n[0]} new -> {out_path}")

async def main_async(args):
    dataset_name = args.dataset_name or Path(args.parquet).stem   
    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    if args.n_samples and args.n_samples < len(df):
        df = df.head(args.n_samples)
    log.info(f"dataset='{dataset_name}'  loaded {len(df)} samples from {args.parquet}")

    
    out_dir = Path(args.out_dir) / dataset_name / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {"dataset": dataset_name, "parquet": str(args.parquet), "n_samples": len(df),
            "models": args.models, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "prompt": "CoT (COT_SYSTEM + ANSWER_FORMAT[mcq/open/binary] + OUTPUT_SCHEMA; step-by-step→'Answer:'; format in user msg)",
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log.info(f"outputs -> {out_dir}")
    write_lock = asyncio.Lock()

    await asyncio.gather(*[
        run_one_model(key, df, args.sid_col, args.img_col, args.q_col,
                      out_dir / f"answers__{key}.jsonl",
                      args.max_tokens, args.temperature, args.max_concurrency, write_lock,
                      dataset_name, args.seed, args.top_p, args.answer_mode)
        for key in args.models
    ])
    log.info("ALL MODELS DONE.")

def main():
    ap = argparse.ArgumentParser(description="MME-CoT opt 5-model answer generation (vLLM async)")
    ap.add_argument("--parquet", required=True,
                    help="입력 parquet (예: data/MMStar_converted.parquet)")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    help=f"which models to run (keys: {list(MODELS)})")
    ap.add_argument("--out-dir", default="mme_outputs")
    ap.add_argument("--run-id", default="opt_v1")
    ap.add_argument("--dataset-name", default=None,
                    help="데이터셋 식별명(레코드/경로에 기록). 기본=parquet 파일명")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42, help="샘플링 재현용 seed(모든 요청 공통)")
    ap.add_argument("--max-concurrency", type=int, default=0,
                    help="모델당 in-flight 상한(0=레지스트리 기본값). 서버 --max-num-seqs 와 맞추세요.")
    ap.add_argument("--n-samples", type=int, default=0, help="빠른 점검용 상위 N개(0=전체)")
    ap.add_argument("--sid-col", default="index")
    ap.add_argument("--img-col", default="image")
    ap.add_argument("--q-col", default="question")
    ap.add_argument("--answer-mode", choices=["auto", "mcq", "open", "binary"], default="auto",
                    help="auto=보기유무로 mcq/open, binary=yes/no(HallusionBench)")
    args = ap.parse_args()
    bad = [m for m in args.models if m not in MODELS]
    if bad:
        sys.exit(f"unknown model keys: {bad}; valid={list(MODELS)}")
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
