#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA  # noqa: E402
from _stats import breslow_day, mh_or, standardize  # noqa: E402

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPECS = [
    ("hb", "HallusionBench VD vs VS", "VD", "VS", "cat", "MaC",
     lambda g: g.f < g.b, "MaC loss"),
    ("hb", "HallusionBench VD vs VS", "VD", "VS", "cat", "MiC",
     lambda g: g.f > g.b, "MiC recovery"),
    ("ms", "MMStar perception vs reasoning", "perception", "reasoning", "grp", "MaC",
     lambda g: g.f < g.b, "MaC loss"),
    ("ms", "MMStar perception vs reasoning", "perception", "reasoning", "grp", "MiC",
     lambda g: g.f > g.b, "MiC recovery"),
    ("ms", "MMStar perception vs reasoning", "perception", "reasoning", "grp", "CC",
     lambda g: g.f < g.b, "CC loss"),
]

def analyze(P, axis, expo, base, col, sit, event, name):
    G = P[(P.situation == sit) & (P[col].isin([expo, base]))].copy()
    if not len(G):
        return []
    G["event"] = event(G).astype(int)
    G["stratum"] = pd.qcut(G.wi_lto.rank(method="first"), 5,
                           labels=[f"S{i+1}" for i in range(5)])
    tabs, rows = [], []
    for s, g in G.groupby("stratum", observed=True):
        e, b = g[g[col] == expo], g[g[col] == base]
        A, B_ = int(e.event.sum()), int((1 - e.event).sum())
        C, D = int(b.event.sum()), int((1 - b.event).sum())
        if min(A + B_, C + D) == 0:
            continue
        tabs.append((A, B_, C, D))
        rows.append(dict(axis=axis, situation=sit, event=name, stratum=str(s),
                         difficulty=float(g.wi_lto.mean()),
                         n_expo=A + B_, rate_expo=A / (A + B_),
                         n_base=C + D, rate_base=C / (C + D),
                         OR=(A * D) / (B_ * C) if B_ * C else np.nan))
    if not tabs:
        return []
    o, lo, hi = mh_or(tabs)
    bd, df_, p = breslow_day(tabs, o)

    w = G[G[col] == expo].groupby("stratum", observed=True).size()
    w = (w / w.sum()).to_dict()
    r = G[G[col] == base].groupby("stratum", observed=True).event.mean().to_dict()
    std = standardize(r, w)
    raw_e = float(G[G[col] == expo].event.mean())
    raw_b = float(G[G[col] == base].event.mean())

    rows.append(dict(axis=axis, situation=sit, event=name, stratum="ALL",
                     difficulty=float(G.wi_lto.mean()),
                     n_expo=int((G[col] == expo).sum()), rate_expo=raw_e,
                     n_base=int((G[col] == base).sum()), rate_base=raw_b,
                     OR=np.nan, OR_MH=o, lo=lo, hi=hi,
                     BD_chi2=bd, BD_df=df_, BD_p=p,
                     rate_base_std=std, diff_raw=raw_e - raw_b,
                     diff_std=raw_e - std))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()

    HB = pd.read_csv(os.path.join(a.data, "visual_panel_hb.csv"))
    MS = pd.read_csv(os.path.join(a.data, "visual_panel_mmstar.csv"))
    MS["grp"] = MS.grp.map({"지각": "perception", "추론": "reasoning"}).fillna(MS.grp)
    print(f"[입력] {a.data}/visual_panel_hb.csv ({len(HB):,}행), "
          f"visual_panel_mmstar.csv ({len(MS):,}행)")
    print("[층화] wi_lto 5분위 · [단위] (팀, 문항)\n")

    rows = []
    for panel, axis, e, b, col, sit, ev, name in SPECS:
        rows += analyze(HB if panel == "hb" else MS, axis, e, b, col, sit, ev, name)
    R = pd.DataFrame(rows)
    out = os.path.join(a.data, "supC_visual_robustness.csv")
    R.to_csv(out, index=False, encoding="utf-8-sig")

    S = R[R.stratum == "ALL"]
    print("[C-1] Mantel--Haenszel 공통 오즈비와 Breslow--Day 동질성")
    print(S[["axis", "situation", "event", "OR_MH", "lo", "hi",
             "BD_chi2", "BD_df", "BD_p"]].round(4).to_string(index=False))
    print("\n  BD_p 가 크면 층 간 오즈비가 다르다는 증거가 없다는 뜻 "
          "— 공통 오즈비 요약이 정당해진다.")

    print("\n[C-2] 직접 표준화 — 대조군을 노출군의 난이도 분포로 재가중")
    print(S[["axis", "situation", "event", "rate_expo", "rate_base",
             "rate_base_std", "diff_raw", "diff_std"]].round(4).to_string(index=False))
    print("\n  diff_std 가 diff_raw 와 비슷하게 남으면 난이도 구성 차이로 설명되지 않는다.")

    print("\n[C-3] 층별 수치 (Figure 3 의 값)")
    print(R[R.stratum != "ALL"][["axis", "situation", "event", "stratum",
                                 "n_expo", "rate_expo", "n_base", "rate_base", "OR"]]
          .round(4).to_string(index=False))
    print(f"\n[저장] {out}")

if __name__ == "__main__":
    main()
