#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import os
import argparse
from collections import Counter
from math import erfc, log, sqrt, exp

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stats import breslow_day  # noqa: E402

RMAX = int(os.environ.get("RLAST", "3"))   # C-03: R_last

def chi2_sf(x, k=1):
    if k != 1:
        raise ValueError
    return erfc(sqrt(max(x, 0.0) / 2.0))

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
    hb = df[(df.benchmark == "hallusionbench")].copy()
    d = hb[hb["mode"] == "debate"]
    wi = d[d["round"] == 0].groupby(["question_id", "model"]).is_correct.mean().unstack()
    models = list(wi.columns)

    rows, excl = [], 0
    for (t, q), g in d.groupby(["team_id", "question_id"]):
        gold = g.gold.iloc[0]
        r0, rl = vote(list(g[g["round"] == 0].answer)), vote(list(g[g["round"] == RMAX].answer))
        if r0 is None or rl is None:
            excl += 1
            continue
        v0, c0, mx0, t0 = r0
        _, _, _, tl = rl
        mem = set(g.model)
        out = [m for m in models if m not in mem]
        rows.append(dict(cat=g.category.iloc[0], team_id=t, question_id=q,
                         situation=classify(c0, mx0, t0, gold),
                         b=(1 / len(t0)) if gold in t0 else 0.0,
                         f=(1 / len(tl)) if gold in tl else 0.0,
                         wi_lto=float(wi.loc[q, out].mean()),
                         wi_all=float(wi.loc[q].mean())))
    P = pd.DataFrame(rows)
    P["delta"] = P.f - P.b
    return P, excl

def mh(tabs):
    num = sum(a * d / (a + b + c + d) for a, b, c, d in tabs)
    den = sum(b * c / (a + b + c + d) for a, b, c, d in tabs)
    if den == 0 or num == 0:
        return np.nan, (np.nan, np.nan), np.nan
    or_mh = num / den
    
    P = [(a + d) / (a + b + c + d) for a, b, c, d in tabs]
    Q = [(b + c) / (a + b + c + d) for a, b, c, d in tabs]
    R = [a * d / (a + b + c + d) for a, b, c, d in tabs]
    S = [b * c / (a + b + c + d) for a, b, c, d in tabs]
    sR, sS = sum(R), sum(S)
    v = (sum(p * r for p, r in zip(P, R)) / (2 * sR ** 2)
         + sum(p * s + q * r for p, q, r, s in zip(P, Q, R, S)) / (2 * sR * sS)
         + sum(q * s for q, s in zip(Q, S)) / (2 * sS ** 2))
    se = sqrt(v)
    lo, hi = exp(log(or_mh) - 1.96 * se), exp(log(or_mh) + 1.96 * se)
    
    E = sum((a + b) * (a + c) / (a + b + c + d) for a, b, c, d in tabs)
    O = sum(a for a, b, c, d in tabs)
    V = sum((a + b) * (c + d) * (a + c) * (b + d) / ((a + b + c + d) ** 2 * (a + b + c + d - 1))
            for a, b, c, d in tabs if (a + b + c + d) > 1)
    chi = (abs(O - E) - 0.5) ** 2 / V if V > 0 else np.nan
    return or_mh, (lo, hi), chi2_sf(chi)

def analyze(P, situation, event_col, event_name, dcol="wi_lto"):
    G = P[P.situation == situation].copy()
    G["event"] = event_col(G)
    G = G.rename(columns={dcol: "_d"})
    print(f"\n{'='*88}\n[{situation}] {event_name}\n{'='*88}")

    
    print(f"  난이도({dcol}) 분포")
    for c in ["VD", "VS"]:
        g = G[G.cat == c]
        print(f"    {c}: n={len(g):5,}  평균={g._d.mean():.4f}  "
              f"사분위={g._d.quantile([.25,.5,.75]).round(3).tolist()}")

    
    a = int(((G.cat == "VD") & (G.event == 1)).sum()); b = int(((G.cat == "VD") & (G.event == 0)).sum())
    c_ = int(((G.cat == "VS") & (G.event == 1)).sum()); d = int(((G.cat == "VS") & (G.event == 0)).sum())
    crude = (a * d) / (b * c_) if b * c_ else np.nan
    print(f"\n  조야한(층화 없음) 비교")
    print(f"    VD {a/(a+b):.4f} ({a}/{a+b})   VS {c_/(c_+d):.4f} ({c_}/{c_+d})   "
          f"차이 {a/(a+b)-c_/(c_+d):+.4f}   OR={crude:.3f}")

    
    q = pd.qcut(G._d.rank(method="first"), 5, labels=[f"S{i+1}" for i in range(5)])
    G["stratum"] = q
    tabs, rowout = [], []
    for s, g in G.groupby("stratum", observed=True):
        vd, vs = g[g.cat == "VD"], g[g.cat == "VS"]
        A, B = int(vd.event.sum()), int((1 - vd.event).sum())
        C, D = int(vs.event.sum()), int((1 - vs.event).sum())
        if min(A + B, C + D) == 0:
            continue
        tabs.append((A, B, C, D))
        rowout.append(dict(층=s, 난이도=g._d.mean(), n_VD=A + B, VD=A / (A + B) if A + B else np.nan,
                           n_VS=C + D, VS=C / (C + D) if C + D else np.nan,
                           차이=(A / (A + B) - C / (C + D)) if (A + B) and (C + D) else np.nan,
                           OR=(A * D) / (B * C) if B * C else np.nan))
    print(f"\n  난이도 5분위 층화")
    print("   " + pd.DataFrame(rowout).round(4).to_string(index=False).replace("\n", "\n   "))

    or_mh, (lo, hi), p = mh(tabs)
    bd, dfree, pbd = breslow_day(tabs, or_mh)
    print(f"\n  Mantel-Haenszel 공통 OR = {or_mh:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p={p:.2e}")
    print(f"  조야한 OR = {crude:.3f}  ->  층화 후 {or_mh:.3f}  "
          f"(난이도로 설명되는 부분: {100*(1-(or_mh-1)/(crude-1)):.0f}%)" if crude > 1 and or_mh > 1 else
          f"  조야한 OR = {crude:.3f}  ->  층화 후 {or_mh:.3f}")
    print(f"  Breslow--Day 동질성 chi2={bd:.2f} df={dfree} p={pbd:.4f}")

    
    w = G[G.cat == "VD"].groupby("stratum", observed=True).size()
    w = w / w.sum()
    vs_rate = G[G.cat == "VS"].groupby("stratum", observed=True).event.mean()
    std = float((w * vs_rate).dropna().sum() / w[vs_rate.notna()].sum())
    vd_rate = float(G[G.cat == "VD"].event.mean())
    print(f"\n  직접 표준화 (VS 를 VD 난이도 분포로) : VD {vd_rate:.4f}  vs  VS_표준화 {std:.4f}  "
          f"차이 {vd_rate-std:+.4f}   (표준화 전 차이 {vd_rate - float(G[G.cat=='VS'].event.mean()):+.4f})")
    return G

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
    print(f"[필터] hallusionbench + debate | (팀,문항) {len(P):,}개 | 제외(유효답<3) {excl}개")
    print(f"[확인] f 의 고유값 = {sorted(P.f.unique())}  -> 이진이라 2x2 분할표 사용 가능")

    analyze(P, "MaC", lambda g: (g.f == 0).astype(int), "손실 발생률 (R0 다수 정답 -> R4 오답)")
    analyze(P, "MiC", lambda g: (g.f == 1).astype(int), "구제 발생률 (R0 소수 정답 -> R4 정답)")
    print("\n\n" + "#" * 88)
    print("# Sensitivity: use wi_all difficulty from five models and 30 draws")
    print("#" * 88)
    analyze(P, "MaC", lambda g: (g.f == 0).astype(int), "손실 발생률", dcol="wi_all")
    analyze(P, "MiC", lambda g: (g.f == 1).astype(int), "구제 발생률", dcol="wi_all")
    P.to_csv(os.path.join(a.out, "visual_panel_hb.csv"), index=False)
    print(f"\n[저장] {a.out}/visual_panel_hb.csv")

if __name__ == "__main__":
    main()
