

from __future__ import annotations
import argparse, asyncio, glob, importlib.util, json, os
from pathlib import Path
from openai import AsyncOpenAI

_spec = importlib.util.spec_from_file_location("slj", str(Path(__file__).parent / "score_llm_judge.py"))
slj = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(slj)
normalize_answer = slj.normalize_answer
_gold_answer_variants = slj._gold_answer_variants
NO_ANSWER = slj.NO_ANSWER
TAIL_CHARS = slj.TAIL_CHARS

_EQUIV_SYSTEM = (
    "You grade whether a model's answer to a visual question is CORRECT by comparing it to "
    "the gold answer. They are CORRECT (equivalent) if they denote the SAME value or meaning, "
    "allowing different notation, units, formatting, rounding (~2-3 significant figures), "
    "fraction/decimal forms, or equivalent phrasing. If the model gives only a symbolic "
    "expression while the gold is a specific number (or vice versa), and they cannot be seen "
    "to be equal from the text alone, answer 'no'. Output ONLY 'yes' or 'no'."
)
_EQUIV_EXAMPLES = [
    ("Gold: 35.7 mm\nModel: 35.7", "yes"),
    ("Gold: $\\frac{8}{3}$\nModel: 8/3", "yes"),
    ("Gold: -5.35\nModel: y = -5.35", "yes"),
    ("Gold: 0.5\nModel: 1/2", "yes"),
    ("Gold: 70.7 cm^3\nModel: 70.69", "yes"),
    ("Gold: 218\nModel: d\\sqrt{m/2qV}", "no"),
    ("Gold: EW\nModel: UD", "no"),
    ("Gold: February 2022\nModel: 2022-02-07", "no"),
]
_EQUIV_USER = "Gold: {gold}\nModel: {pred}\nAre they equivalent? Answer yes or no."

def _equiv_messages(gold, pred):
    msgs = [{"role": "system", "content": _EQUIV_SYSTEM}]
    for ex, ans in _EQUIV_EXAMPLES:
        g, m = ex.split("\nModel: ")
        msgs.append({"role": "user", "content": _EQUIV_USER.format(gold=g.replace("Gold: ", ""), pred=m)})
        msgs.append({"role": "assistant", "content": ans})
    msgs.append({"role": "user", "content": _EQUIV_USER.format(gold=gold, pred=pred)})
    return msgs

class EquivScorer:
    def __init__(self, base, model, max_conc=64):
        self.client = AsyncOpenAI(base_url=base, api_key="EMPTY", max_retries=0, timeout=300)
        self.model = model
        self.extractor = slj.Judge(base, model, max_conc)
        self.sem = asyncio.Semaphore(max_conc)

    async def equiv(self, gold, pred) -> bool:
        kw = {}
        if "qwen" in self.model.lower():
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model, messages=_equiv_messages(gold, pred),
                        max_tokens=4, temperature=0.0, **kw)
                    return (r.choices[0].message.content or "").strip().lower().startswith("y")
                except Exception:
                    await asyncio.sleep(2 ** attempt)
        return False

    async def score_one(self, r):
        resp = r.get("response") or ""
        tail = resp[-TAIL_CHARS:] if resp.strip() else ""
        extracted = await self.extractor.extract(tail)
        no_ans = (extracted.strip().upper() == NO_ANSWER) or (not extracted.strip())
        golds_raw = r.get("gt")
        
        exact = (not no_ans) and (normalize_answer(extracted) in _gold_answer_variants(golds_raw))
        method = "exact" if exact else None
        match = 1.0 if exact else 0.0
        
        if not exact and not no_ans:
            if await self.equiv(str(golds_raw), extracted):
                match, method = 1.0, "equiv"
        return {"sample_id": r["sample_id"], "qtype": r["qtype"], "category": r.get("category"),
                "gt": golds_raw, "extracted": extracted, "no_answer": no_ans,
                "match": match, "match_method": method}

async def main_async(args):
    scorer = EquivScorer(args.judge_base, args.judge_model, args.max_concurrency)
    files = sorted(glob.glob(os.path.join(args.dir, "answers__*.jsonl")))
    print(f"extractor+equiv judge: {args.judge_model} @ {args.judge_base}\n" + "=" * 84)
    print(f"{'model':<14} {'MCQ':>13} {'OPEN':>13} {'전체':>13} {'등가회수':>8} {'no_ans':>7}")
    print("-" * 84)
    for f in files:
        k = os.path.basename(f).replace("answers__", "").replace(".jsonl", "")
        rows = [json.loads(l) for l in open(f)]
        res = await asyncio.gather(*[scorer.score_one(r) for r in rows])
        with open(os.path.join(args.dir, f"equiv__{k}.jsonl"), "w") as g:
            for x in res:
                g.write(json.dumps(x, ensure_ascii=False) + "\n")
        def acc(sub):
            n = len(sub); c = sum(x["match"] for x in sub)
            return f"{int(c)}/{n} ({c/max(n,1)*100:.0f}%)"
        mcq = [x for x in res if x["qtype"] == "mcq"]; opn = [x for x in res if x["qtype"] == "open"]
        equiv_n = sum(x["match_method"] == "equiv" for x in res)
        na = sum(x["no_answer"] for x in res)
        print(f"{k:<14} {acc(mcq):>13} {acc(opn):>13} {acc(res):>13} {equiv_n:>8} {na:>7}")
    print("\n판정 상세: equiv__<model>.jsonl (extracted, match, match_method[exact|equiv])")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mme_outputs/MME-CoT_opt/run_cot")
    ap.add_argument("--judge-base", default="http://localhost:8003/v1")
    ap.add_argument("--judge-model", default="gemma-4-12B-it")
    ap.add_argument("--max-concurrency", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))

if __name__ == "__main__":
    main()
