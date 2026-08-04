#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import argparse
import os
from collections import Counter

import pandas as pd

RMAX = int(os.environ.get("RLAST", "3"))          # C-03: R_last

GRP = {
    "coarse perception": "지각",
    "fine-grained perception": "지각",
    "logical reasoning": "추론",
    "math": "추론",
    "instance reasoning": "중간",
    "science & technology": "중간",
}

def vote(a):
    v = [x for x in a if x]
    if len(v) < 3:
        return None
    c = Counter(v)
    mx = max(c.values())
    return v, c, mx, {k for k, n in c.items() if n == mx}

def classify(c, mx, top, g):
    if mx == 3:
        return "MaC" if g in top else "MaW"
    if mx == 2:
        return "MaC" if g in top else ("MiC" if g in c else "MaW")
    return "CC" if g in c else "CW"

def build(df):
    ms = df[df.benchmark == "mmstar"].copy()
    d = ms[ms["mode"] == "debate"]

    
    wi = d[d["round"] == 0].groupby(["question_id", "model"]).is_correct.mean().unstack()
    models = list(wi.columns)

    rows, excl = [], 0
    for (t, q), g in d.groupby(["team_id", "question_id"]):
        gold = g.gold.iloc[0]
        r0 = vote(list(g[g["round"] == 0].answer))
        rl = vote(list(g[g["round"] == RMAX].answer))
        if r0 is None or rl is None:
            excl += 1
            continue
        _, c0, mx0, t0 = r0
        _, _, _, tl = rl
        out = [m for m in models if m not in set(g.model)]     
        cat = g.category.iloc[0]
        rows.append(dict(cat=cat, team_id=t, question_id=q,
                         situation=classify(c0, mx0, t0, gold),
                         b=(1 / len(t0)) if gold in t0 else 0.0,
                         f=(1 / len(tl)) if gold in tl else 0.0,
                         wi_lto=float(wi.loc[q, out].mean()),
                         grp=GRP.get(cat, "중간")))
    P = pd.DataFrame(rows)
    P["delta"] = P.f - P.b
    return P[["cat", "team_id", "question_id", "situation",
              "b", "f", "wi_lto", "delta", "grp"]], excl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["round"] = df["round"].astype(int)
    df["is_correct"] = df["is_correct"].astype(int)

    P, excl = build(df)
    print(f"[입력] {a.csv}")
    print(f"[필터] mmstar + debate | (팀,문항) {len(P):,}개 | 제외(유효답<3) {excl}개 | R_last=R{RMAX}")
    print(f"[축]   " + " / ".join(f"{k} {v:,}" for k, v in P.grp.value_counts().items()))
    print("\n[상황 x 축] (중간 포함)")
    print(pd.crosstab(P.situation, P.grp).to_string())

    P.to_csv(os.path.join(a.out, "visual_panel_mmstar.csv"), index=False)
    print(f"\n[저장] {a.out}/visual_panel_mmstar.csv")

if __name__ == "__main__":
    main()
