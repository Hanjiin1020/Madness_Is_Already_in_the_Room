

from __future__ import annotations
import argparse, ast, asyncio, glob, json, os, re
from collections import defaultdict
import numpy as np
from openai import AsyncOpenAI

TAIL_CHARS = 600
NO_ANSWER = "NONE"

# ── normalize_answer + _UNIT_ALIASES ──────────────────────────────────────────
_UNIT_ALIASES = {
    r"\bkm/h\b": "km/h", r"\bkph\b": "km/h", r"\bmph\b": "mph",
    r"\bsq\.?\s*m\b": "sq m", r"\bm\^2\b": "sq m",
}

def normalize_answer(raw: str) -> str:
    s = str(raw).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    m = re.match(r'^\(([a-z])\)(.*)$', s)
    if m:
        s = m.group(1)
    for pattern, replacement in _UNIT_ALIASES.items():
        s = re.sub(pattern, replacement, s)
    return s

def _gold_answer_variants(gold_answer) -> list[str]:
    if isinstance(gold_answer, str):
        s = gold_answer.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                gold_answer = parsed
    if isinstance(gold_answer, (list, tuple, np.ndarray)):
        return [normalize_answer(str(g)) for g in gold_answer]
    return [normalize_answer(str(gold_answer))]

_ANSWER_SYSTEM = (
    "You extract the FINAL ANSWER from the concluding line(s) of a model's solution "
    "to a visual question.\n"
    "Output ONLY the final answer itself — no explanation, no label, no prefix, "
    "no surrounding punctuation or markdown.\n"
    "- If the answer is a multiple-choice selection, output ONLY the single option "
    "letter in uppercase (e.g. 'D').\n"
    "- Otherwise output the answer value exactly as stated "
    "(a number, expression, word, or short phrase).\n"
    "- If the concluding text states no answer, is undecided, or contains nothing "
    "that can be extracted as an answer, output exactly 'NONE'."
)
_ANSWER_EXAMPLES = [
    ("Final answer is (D) awe.", "D"),
    ("G", "G"),
    ("Looking at the options, the correct option is **G**.", "G"),
    ("**Answer:** (D) amaryllis flower", "D"),
    ("*Answer*: H", "H"),
    ("Therefore, the correct answer is C.", "C"),
    ("Final answer is $\\frac{9}{10}$", "$\\frac{9}{10}$"),
    ("Thus the total area is 2500.", "2500"),
    ("So, there is no answer in this option.", "NONE"),
    ("I cannot determine the answer from the image.", "NONE"),
]
_ANSWER_USER_TMPL = "Extract the final answer from the following concluding text:\n{tail}"

def _extract_messages(tail: str) -> list[dict]:
    msgs = [{"role": "system", "content": _ANSWER_SYSTEM}]
    for ex_text, ex_ans in _ANSWER_EXAMPLES:
        msgs.append({"role": "user", "content": _ANSWER_USER_TMPL.format(tail=ex_text)})
        msgs.append({"role": "assistant", "content": ex_ans})
    msgs.append({"role": "user", "content": _ANSWER_USER_TMPL.format(tail=tail)})
    return msgs

class Judge:
    def __init__(self, base, model, max_conc=64):
        self.client = AsyncOpenAI(base_url=base, api_key="EMPTY", max_retries=0, timeout=300)
        self.model = model
        self.sem = asyncio.Semaphore(max_conc)

    async def extract(self, tail: str) -> str:
        if not tail or not tail.strip():
            return NO_ANSWER
        kw = {}
        if "qwen" in self.model.lower():   
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model, messages=_extract_messages(tail),
                        max_tokens=32, temperature=0.0, **kw)
                    a = (r.choices[0].message.content or "").strip()
                    return a if a else NO_ANSWER
                except Exception:
                    await asyncio.sleep(2 ** attempt)
        return NO_ANSWER

async def score_file(judge, path, out_path):
    rows = [json.loads(l) for l in open(path)]
    async def one(r):
        resp = r.get("response") or ""
        tail = resp[-TAIL_CHARS:] if resp.strip() else ""
        extracted = await judge.extract(tail)
        no_ans = (extracted.strip().upper() == NO_ANSWER) or (not extracted.strip())
        match = 0.0 if no_ans else float(normalize_answer(extracted) in _gold_answer_variants(r.get("gt")))
        return {**{k: r[k] for k in ("sample_id", "qtype", "category", "gt")},
                "extracted": extracted, "no_answer": no_ans, "match": match}
    res = await asyncio.gather(*[one(r) for r in rows])
    with open(out_path, "w") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return res

async def main_async(args):
    judge = Judge(args.judge_base, args.judge_model, args.max_concurrency)
    files = sorted(glob.glob(os.path.join(args.dir, "answers__*.jsonl")))
    print(f"judge: {args.judge_model} @ {args.judge_base}\n" + "=" * 78)
    print(f"{'model':<14} {'MCQ':>13} {'OPEN':>13} {'전체':>13} {'no_answer':>10}")
    print("-" * 78)
    for f in files:
        k = os.path.basename(f).replace("answers__", "").replace(".jsonl", "")
        res = await score_file(judge, f, os.path.join(args.dir, f"judge__{k}.jsonl"))
        def acc(sub):
            n = len(sub); c = sum(x["match"] for x in sub)
            return f"{int(c)}/{n} ({c/max(n,1)*100:.0f}%)"
        mcq = [x for x in res if x["qtype"] == "mcq"]; opn = [x for x in res if x["qtype"] == "open"]
        na = sum(x["no_answer"] for x in res)
        print(f"{k:<14} {acc(mcq):>13} {acc(opn):>13} {acc(res):>13} {f'{na}/{len(res)}':>10}")
    print("\n판정 상세: judge__<model>.jsonl (sample_id, gt, extracted, match, no_answer)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mme_outputs/MME-CoT_opt/run_cot")
    ap.add_argument("--judge-base", default="http://localhost:8003/v1")
    ap.add_argument("--judge-model", default="gemma-4-12B-it")
    ap.add_argument("--max-concurrency", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))

if __name__ == "__main__":
    main()
