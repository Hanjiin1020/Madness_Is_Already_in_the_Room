#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import (  # noqa: E402
    plt, save, pv, PT, W1, BM, LBL, SHORT, SIT,
    DIV5, H5, CORRECT, WRONG, TINT_C, TINT_W, GREYD, DARKFILL)

def fig1(D, out):
    T0 = pd.read_csv(os.path.join(D, "t3f1_by_state.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(W1, 2.45))
    fig.subplots_adjust(left=.17, right=.99, bottom=.30, top=.97)

    d = pv(T0, "benchmark", "situation", "delta평균")
    lo = pv(T0, "benchmark", "situation", "delta_lo")
    hi = pv(T0, "benchmark", "situation", "delta_hi")

    x = np.arange(len(BM))
    w = .165
    for i, s in enumerate(SIT):
        v = d[s].values
        err = np.abs(np.vstack([v - lo[s].values, hi[s].values - v]))
        ax.bar(x + (i - 2) * w, v, w, yerr=err, capsize=1.5,
               color=DIV5[i], hatch=H5[i], edgecolor="black", linewidth=.6,
               error_kw=dict(lw=.7), label=s)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[b] for b in BM], fontsize=PT)
    ax.set_ylabel(r"Mean $\Delta = f - b$")
    ax.grid(axis="y", linestyle=":", alpha=.6)
    ax.set_axisbelow(True)

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=PT, frameon=False, loc="lower center",
               bbox_to_anchor=(.56, .00), columnspacing=1.1,
               handlelength=1.3, handletextpad=.4)
    save(fig, out, "fig1_configuration_gain_loss", target_w=W1)

def fig2(D, out):
    M = pd.read_csv(os.path.join(D, "f2_adoption.csv"), encoding="utf-8-sig")
    Bl = pd.read_csv(os.path.join(D, "f2_adoption_baseline.csv"), encoding="utf-8-sig")

    keys = ["채택_정답", "채택_오답", "창발_정답", "창발_오답"]
    names = ["Adopted, correct", "Adopted, wrong",
             "Generated, correct", "Generated, wrong"]
    cols = [CORRECT, WRONG, TINT_C, TINT_W]
    hats = ["", "", "xx", "xx"]

    P = M.pivot(index="benchmark", columns="지표", values="값").reindex(BM)[keys].values
    Bl = Bl.set_index("benchmark").reindex(BM)
    base = Bl.우연채택기준선.values * 100
    excess = Bl.초과.values * 100

    fig, ax = plt.subplots(figsize=(W1, 2.60))
    fig.subplots_adjust(left=.27, right=.98, bottom=.40, top=.97)

    y = np.arange(len(BM))[::-1]
    left = np.zeros(len(BM))
    for i in range(4):
        v = np.nan_to_num(P[:, i]) * 100
        ax.barh(y, v, left=left, height=.48, color=cols[i], hatch=hats[i],
                edgecolor="black", linewidth=.6, label=names[i])
        for yy, ll, vv in zip(y, left, v):
            if vv >= 12:
                ax.text(ll + vv / 2, yy, f"{vv:.0f}", ha="center", va="center",
                        fontsize=PT,
                        color="white" if cols[i] in DARKFILL else "black")
        left += v

    
    tot = P[:, :2].sum(axis=1) * 100
    for yy, b, e, tt in zip(y, base, excess, tot):
        ax.plot([b, b], [yy - .24, yy + .24], color="black", lw=1.3,
                solid_capstyle="butt", zorder=5)
        ax.annotate("", xy=(tt, yy + .40), xytext=(b, yy + .40),
                    arrowprops=dict(arrowstyle="|-|,widthA=.28,widthB=.28",
                                    lw=.8, color="black", shrinkA=0, shrinkB=0),
                    zorder=5)
        ax.text((b + tt) / 2, yy + .40, f"$+${e:.0f}", ha="center", va="bottom",
                fontsize=PT, zorder=6,
                bbox=dict(boxstyle="square,pad=.06", fc="white", ec="none"))

    ax.set_yticks(y)
    ax.set_yticklabels([LBL[b] for b in BM], fontsize=PT)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(y.min() - .50, y.max() + .78)
    ax.set_xlabel("Share of team\u2013questions (%)")

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=PT, frameon=False, loc="lower center",
               bbox_to_anchor=(.54, .01), columnspacing=1.0,
               handlelength=1.3, handletextpad=.4, labelspacing=.30)
    save(fig, out, "fig2_adoption_generation", target_w=W1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=FIGS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.data, a.out)
    fig2(a.data, a.out)
