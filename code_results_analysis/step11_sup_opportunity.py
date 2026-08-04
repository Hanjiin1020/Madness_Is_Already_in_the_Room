#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA  # noqa: E402

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
LBL = {"hallusionbench": "HallusionBench", "mmstar": "MMStar",
       "mmmu": "MMMU", "mme-cot": "MME-CoT"}
HAVE = {"MaC", "MiC", "CC"}      
ROOM = {"MiC", "CC"}             
NONE = {"MaW", "CW"}             
B, SEED = 2000, 42

def stats(g):
    have = g.situation.isin(HAVE)
    room = g.situation.isin(ROOM)
    none = g.situation.isin(NONE)
    oracle = float(have.mean())
    b, f = float(g.b.mean()), float(g.f.mean())
    gap = oracle - b
    return {
        "oracle_r0": oracle,
        "b": b,
        "f": f,
        "gain": f - b,
        "oracle_gap_recovery": (f - b) / gap if gap > 1e-12 else np.nan,
        "f_given_room": float(g.f[room].mean()) if room.any() else np.nan,
        "f_given_none": float(g.f[none].mean()) if none.any() else np.nan,
    }

def qboot(g, keys):
    idx = {q: np.flatnonzero(g.question_id.values == q)
           for q in g.question_id.unique()}
    qs = np.array(list(idx))
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in keys}
    for _ in range(B):
        pick = rng.choice(qs, size=len(qs), replace=True)
        rows = np.concatenate([idx[q] for q in pick])
        s = stats(g.iloc[rows])
        for k in keys:
            draws[k].append(s[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in draws.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()

    src = os.path.join(a.data, "panel_team_question.csv.gz")
    P = pd.read_csv(src)
    print(f"[입력] {src}   팀-문항 {len(P):,}행")
    print(f"[규칙] R_last=R3 · b,f = 1[gold∈T]/|T| · 부트스트랩 B={B} seed={SEED}\n")

    keys = list(stats(P[P.benchmark == BM[0]]))
    rows = []
    for bm in BM:
        g = P[P.benchmark == bm]
        s = stats(g)
        ci = qboot(g, keys)
        for k in keys:
            rows.append(dict(benchmark=bm, 지표=k, n=len(g), 값=s[k],
                             lo=ci[k][0], hi=ci[k][1]))
    R = pd.DataFrame(rows)
    out = os.path.join(a.data, "supA_opportunity.csv")
    R.to_csv(out, index=False, encoding="utf-8-sig")

    piv = R.pivot(index="benchmark", columns="지표", values="값").reindex(BM)
    show = piv[["oracle_r0", "b", "f", "gain", "oracle_gap_recovery",
                "f_given_room", "f_given_none"]]
    show.index = [LBL[b] for b in show.index]
    print("[결과]")
    print(show.round(4).to_string())
    print("\n  oracle_r0            라운드 0 답 풀에 정답이 있는 비율")
    print("  oracle_gap_recovery  (f−b)/(oracle−b) — 여유분 중 회수한 몫")
    print("  f_given_room         MiC·CC(정답 있으나 다수결 아님)에서의 도달률")
    print("  f_given_none         MaW·CW(정답 없음)에서의 도달률 = 생성 성공률")

    nq = P.groupby("benchmark").apply(lambda g: g.situation.value_counts(),
                                      include_groups=False).unstack()
    print("\n[상태별 팀-문항 수]")
    print(nq.reindex(BM).fillna(0).astype(int).to_string())
    print(f"\n[저장] {out}")

if __name__ == "__main__":
    main()
