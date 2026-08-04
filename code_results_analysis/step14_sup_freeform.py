#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, ROOT  # noqa: E402

import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B, SEED = 2000, 42

_UNIT_ALIASES = {
    r"\bkm/h\b": "km/h", r"\bkph\b": "km/h", r"\bmph\b": "mph",
    r"\bsq\.?\s*m\b": "sq m", r"\bm\^2\b": "sq m",
}

def normalize_answer(raw):
    s = str(raw).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    m = re.match(r"^\(([a-z])\)(.*)$", s)
    if m:
        s = m.group(1)
    for pat, rep in _UNIT_ALIASES.items():
        s = re.sub(pat, rep, s)
    return s

def gold_variants(gold):
    if isinstance(gold, str):
        s = gold.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                p = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                p = None
            if isinstance(p, (list, tuple)):
                gold = p
    if isinstance(gold, (list, tuple)):
        return [normalize_answer(str(g)) for g in gold]
    return [normalize_answer(str(gold))]

def cache_stats(path):
    c = json.load(open(path, encoding="utf-8"))
    ex_t = ex_f = j_y = j_n = 0
    for k, v in c.items():
        gold, pred = k.split("\t", 1)
        if normalize_answer(pred) in gold_variants(gold):
            ex_t += bool(v)
            ex_f += (not v)
        else:
            j_y += bool(v)
            j_n += (not v)
    return dict(pairs_total=len(c), exact_true=ex_t, exact_false=ex_f,
                judged=j_y + j_n, judge_yes=j_y, judge_no=j_n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--cache",
                    default=os.path.join(ROOT, "code_judge", "equiv_cache.json"))
    a = ap.parse_args()

    S = cache_stats(a.cache)
    print(f"[입력] {a.cache}")
    print(f"[캐시] 유니크 쌍 {S['pairs_total']:,}  "
          f"| exact 확정 {S['exact_true'] + S['exact_false']:,} "
          f"| judge 판정 {S['judged']:,} → 등가 {S['judge_yes']:,} "
          f"({S['judge_yes']/S['judged']*100:.1f}%) / 비등가 {S['judge_no']:,}\n")

    d = pd.read_csv(a.csv, dtype=str, keep_default_na=False,
                    usecols=["benchmark", "question_id", "answer_type", "mode"])
    mc = d[(d.benchmark == "mme-cot") & (d["mode"] == "debate")]
    resp = mc.answer_type.value_counts().to_dict()
    nq = mc.drop_duplicates(["question_id"]).answer_type.value_counts().to_dict()
    print(f"[MME-CoT] 응답 {resp} | 문항 {nq}")

    qt = mc.drop_duplicates(["question_id"]).set_index("question_id").answer_type
    P = pd.read_csv(os.path.join(a.data, "panel_team_question.csv.gz"))
    m = P[P.benchmark == "mme-cot"].copy()
    m["qt"] = m.question_id.astype(str).map(qt)
    g = m.groupby("qt").agg(n=("delta", "size"), b=("b", "mean"), f=("f", "mean"),
                            delta=("delta", "mean"))

    rng = np.random.default_rng(SEED)
    qs = m.question_id.unique()
    idx = {q: np.flatnonzero(m.question_id.values == q) for q in qs}
    dif = []
    for _ in range(B):
        pick = rng.choice(qs, size=len(qs), replace=True)
        s = m.iloc[np.concatenate([idx[q] for q in pick])]
        dif.append(s[s.qt == "mcq"].delta.mean() - s[s.qt == "freeform"].delta.mean())
    d_hat = g.loc["mcq", "delta"] - g.loc["freeform", "delta"]
    lo, hi = np.percentile(dif, 2.5), np.percentile(dif, 97.5)
    print("\n[강건성] MME-CoT 문항 유형별 (팀-문항)")
    print(g.round(4).to_string())
    print(f"  Δ(mcq) − Δ(freeform) = {d_hat:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    rows = [dict(구분="cache", 지표=k, 값=v) for k, v in S.items()]
    rows += [dict(구분="mme-cot", 지표=f"n_resp_{k}", 값=v) for k, v in resp.items()]
    rows += [dict(구분="mme-cot", 지표=f"n_question_{k}", 값=v) for k, v in nq.items()]
    for qtv in ["mcq", "freeform"]:
        for col in ["n", "b", "f", "delta"]:
            rows.append(dict(구분=qtv, 지표=col, 값=float(g.loc[qtv, col])))
    rows += [dict(구분="contrast", 지표="delta_diff", 값=float(d_hat)),
             dict(구분="contrast", 지표="delta_diff_lo", 값=float(lo)),
             dict(구분="contrast", 지표="delta_diff_hi", 값=float(hi))]
    out = os.path.join(a.data, "supD_freeform.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {out}")

if __name__ == "__main__":
    main()
