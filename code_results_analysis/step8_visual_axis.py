#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import argparse
import os
from math import erfc, log, sqrt, exp

import numpy as np
import pandas as pd

def chi2_sf1(x):
    return erfc(sqrt(max(x, 0.0) / 2.0))

def mh(tabs):
    tabs = [t for t in tabs if sum(t) > 1]
    num = sum(a * d / sum((a, b, c, d)) for a, b, c, d in tabs)
    den = sum(b * c / sum((a, b, c, d)) for a, b, c, d in tabs)
    if den == 0 or num == 0:
        return np.nan, np.nan, np.nan, np.nan
    o = num / den
    P = [(a + d) / sum((a, b, c, d)) for a, b, c, d in tabs]
    Q = [(b + c) / sum((a, b, c, d)) for a, b, c, d in tabs]
    R = [a * d / sum((a, b, c, d)) for a, b, c, d in tabs]
    S = [b * c / sum((a, b, c, d)) for a, b, c, d in tabs]
    sR, sS = sum(R), sum(S)
    v = (sum(p * r for p, r in zip(P, R)) / (2 * sR ** 2)
         + sum(p * s + q * r for p, q, r, s in zip(P, Q, R, S)) / (2 * sR * sS)
         + sum(q * s for q, s in zip(Q, S)) / (2 * sS ** 2))
    se = sqrt(v)
    E = sum((a + b) * (a + c) / sum((a, b, c, d)) for a, b, c, d in tabs)
    O = sum(a for a, b, c, d in tabs)
    V = sum((a + b) * (c + d) * (a + c) * (b + d) /
            (sum((a, b, c, d)) ** 2 * (sum((a, b, c, d)) - 1)) for a, b, c, d in tabs)
    p = chi2_sf1((abs(O - E) - 0.5) ** 2 / V) if V > 0 else np.nan
    return o, exp(log(o) - 1.96 * se), exp(log(o) + 1.96 * se), p

def analyze(P, axis, expo_lab, base_lab, catcol, situation, event, ev_name):
    
    G = P[(P.situation == situation) & (P[catcol].isin([expo_lab, base_lab]))].copy()
    if not len(G):
        return None, []
    G["event"] = event(G).astype(int)
    E = G[G[catcol] == expo_lab]
    Bs = G[G[catcol] == base_lab]
    a, b = int(E.event.sum()), int((1 - E.event).sum())
    c, d = int(Bs.event.sum()), int((1 - Bs.event).sum())
    crude = (a * d) / (b * c) if b * c else np.nan
    G["stratum"] = pd.qcut(G.wi_lto.rank(method="first"), 5,
                           labels=[f"S{i+1}" for i in range(5)])
    tabs, per = [], []
    for s, g in G.groupby("stratum", observed=True):
        e, bb = g[g[catcol] == expo_lab], g[g[catcol] == base_lab]
        A, B_ = int(e.event.sum()), int((1 - e.event).sum())
        C, D = int(bb.event.sum()), int((1 - bb.event).sum())
        if min(A + B_, C + D) == 0:
            continue
        tabs.append((A, B_, C, D))
        per.append(dict(axis=axis, situation=situation, event=ev_name, stratum=s,
                        difficulty=float(g.wi_lto.mean()),
                        n_expo=A + B_, rate_expo=A / (A + B_),
                        n_base=C + D, rate_base=C / (C + D),
                        OR=(A * D) / (B_ * C) if B_ * C else np.nan))
    o, lo, hi, p = mh(tabs)
    row = dict(axis=axis, exposed=expo_lab, reference=base_lab,
               situation=situation, event=ev_name,
               n=len(G), n_expo=a + b, n_base=c + d,
               rate_expo=a / (a + b), rate_base=c / (c + d),
               OR_crude=crude, OR_MH=o, lo=lo, hi=hi, p=p)
    return row, per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb", default=os.path.join(DATA, "visual_panel_hb.csv"))
    ap.add_argument("--mmstar",
                default=os.path.join(DATA, "visual_panel_mmstar.csv"))
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    HB = pd.read_csv(a.hb)
    MS = pd.read_csv(a.mmstar)
    MS["grp"] = MS.grp.map({"지각": "perception", "추론": "reasoning"}).fillna(MS.grp)

    rows, per = [], []
    for args_ in [
        
        (HB, "HallusionBench VD vs VS", "VD", "VS", "cat", "MaC",
         lambda g: g.f < g.b, "MaC loss"),
        (HB, "HallusionBench VD vs VS", "VD", "VS", "cat", "MiC",
         lambda g: g.f > g.b, "MiC repair"),
        (HB, "HallusionBench VD vs VS", "VD", "VS", "cat", "MaC",
         lambda g: g.f == 0, "MaC total loss (strict)"),
        (HB, "HallusionBench VD vs VS", "VD", "VS", "cat", "MiC",
         lambda g: g.f == 1, "MiC full repair (strict)"),
        (MS, "MMStar perception vs reasoning", "perception", "reasoning", "grp", "MaC",
         lambda g: g.f < g.b, "MaC loss"),
        (MS, "MMStar perception vs reasoning", "perception", "reasoning", "grp", "MiC",
         lambda g: g.f > g.b, "MiC repair"),
        (MS, "MMStar perception vs reasoning", "perception", "reasoning", "grp", "MaC",
         lambda g: g.f == 0, "MaC total loss (strict)"),
        (MS, "MMStar perception vs reasoning", "perception", "reasoning", "grp", "MiC",
         lambda g: g.f == 1, "MiC full repair (strict)"),
        (MS, "MMStar perception vs reasoning", "perception", "reasoning", "grp", "CC",
         lambda g: g.f < g.b, "CC loss"),
    ]:
        P, axis, e, b, col, sit, ev, name = args_
        r, pr = analyze(P, axis, e, b, col, sit, ev, name)
        if r:
            rows.append(r)
            per += pr
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "s45_mh_or.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(per).to_csv(os.path.join(a.out, "f3_or_by_stratum.csv"),
                             index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))
    print(f"\n[저장] {a.out}/s45_mh_or.csv, {a.out}/f3_or_by_stratum.csv")

if __name__ == "__main__":
    main()
