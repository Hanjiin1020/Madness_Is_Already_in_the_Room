#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import os
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

RMAX = int(os.environ.get("RLAST", "3"))   # C-03: R_last
BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]

def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["round"] = df["round"].astype(int)
    df["is_correct"] = df["is_correct"].astype(int)
    return df[df["mode"] == "debate"].copy()

def valid_keys(d):
    n = (d[d["round"].isin([0, RMAX])]
         .assign(ok=lambda x: (x.answer != "").astype(int))
         .groupby(["benchmark", "team_id", "question_id", "round"]).ok.sum()
         .unstack())
    keep = n[(n[0] == 3) & (n[RMAX] == 3)].index
    return set(keep), len(n) - len(keep)

def error_correlation(d, keys):

    rows = []
    for (b, t), g in d.groupby(["benchmark", "team_id"]):
        g = g[[(b, t, q) in keys for q in g.question_id]]
        for r in range(RMAX + 1):
            piv = (g[g["round"] == r]
                   .pivot_table(index="question_id", columns="model",
                                values="is_correct", aggfunc="first"))
            models = list(piv.columns)
            cors = []
            for i, j in combinations(models, 2):
                sub = piv[[i, j]].dropna()
                ei, ej = 1 - sub[i].values, 1 - sub[j].values
                if len(sub) >= 3 and ei.std() > 1e-9 and ej.std() > 1e-9:
                    cors.append(float(np.corrcoef(ei, ej)[0, 1]))
            rows.append(dict(benchmark=b, team_id=t, round=r, n_q=len(piv),
                             n_pairs=len(cors),
                             err_corr=float(np.mean(cors)) if cors else np.nan))
    return pd.DataFrame(rows)

def wvote(pairs, gold):
    sc = {}
    for a, w in pairs:
        if a:
            sc[a] = sc.get(a, 0.0) + w
    if not sc:
        return np.nan
    mx = max(sc.values())
    top = [a for a, v in sc.items() if v > mx - 1e-12]
    return (1.0 / len(top)) if gold in top else 0.0

def weighted_voting(d, keys, schemes):
    strength = d[d["round"] == 0].groupby(["benchmark", "model"]).is_correct.mean()
    rows = []
    for b, g in d.groupby("benchmark"):
        s = strength[b]
        order = s.sort_values(ascending=False)
        for r in range(RMAX + 1):
            acc = {k: [] for k in ["uniform"] + list(schemes)}
            for (t, q), gg in g[g["round"] == r].groupby(["team_id", "question_id"]):
                if (b, t, q) not in keys:
                    continue
                gold = gg.gold.iloc[0]
                ans = list(gg.answer)
                mems = list(gg.model)
                acc["uniform"].append(wvote(list(zip(ans, [1.0] * len(ans))), gold))
                
                rk = {m: i + 1 for i, m in enumerate(sorted(mems, key=lambda m: -s[m]))}
                for name, fn in schemes.items():
                    w = [fn(s[m], rk[m]) for m in mems]
                    acc[name].append(wvote(list(zip(ans, w)), gold))
            rows.append(dict(benchmark=b, round=r, n=len(acc["uniform"]),
                             **{k: float(np.nanmean(v)) for k, v in acc.items()}))
    return pd.DataFrame(rows), strength

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()

    d = load(a.csv)
    keys, dropped = valid_keys(d)
    print(f"[입력] {a.csv}")
    print(f"[필터] mode=debate | 팀-문항 {len(keys):,}개 사용, {dropped}개 제외(유효답<3)\n")

    # ── A ──
    E = error_correlation(d, keys)
    E.to_csv(os.path.join(a.out, "s46_error_correlation.csv"), index=False)
    piv = E.pivot_table(index="benchmark", columns="round", values="err_corr")
    piv.columns = [f"R{c}" for c in piv.columns]
    piv["R0->Rlast"] = piv[f"R{RMAX}"] - piv["R0"]
    print("[분석 A] 멤버 쌍 오류 상관 (팀 평균)")
    print(piv.loc[BM].round(4).to_string())
    mono = (E.pivot_table(index=["benchmark", "team_id"], columns="round", values="err_corr")
              .diff(axis=1).iloc[:, 1:] > 0).all(axis=1).groupby(level=0).sum()
    print("\n  팀별 단조 증가 (10팀 중):", {b: int(mono[b]) for b in BM})

    # ── B ──
    schemes = {
        "w_acc":      lambda s, rk: s,                    
        "w_acc_sq":   lambda s, rk: s ** 2,               
        "w_rank_321": lambda s, rk: {1: 3, 2: 2, 3: 1}[rk],   
        "w_rank_411": lambda s, rk: {1: 4, 2: 1, 3: 1}[rk],   
    }
    W, strength = weighted_voting(d, keys, schemes)
    W.to_csv(os.path.join(a.out, "extra_weighted_voting.csv"), index=False)
    print("\n[분석 B] 가중 투표 (R0, R4만 표시 — 전체는 extra_weighted_voting.csv)")
    show = W[W["round"].isin([0, RMAX])].set_index(["benchmark", "round"])
    print(show.loc[[(b, r) for b in BM for r in (0, RMAX)]].round(4).to_string())
    print("\n  전역 강도 (R0 정확도):")
    for b in BM:
        s = strength[b].sort_values(ascending=False)
        print(f"    {b:16s} " + "  ".join(f"{m} {v:.3f}" for m, v in s.items()))

if __name__ == "__main__":
    main()
