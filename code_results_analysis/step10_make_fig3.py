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
from _figstyle import plt, save, PT, W1, GREY, DIV5  # noqa: E402

EXPO = DIV5[0]   
                 

AX = {"HallusionBench VD vs VS": "HallusionBench",
      "MMStar perception vs reasoning": "MMStar"}
EV = [("MaC loss", "MaC loss", (0, .155)),
      ("MiC repair", "MiC repair", (.10, .70)),
      ("CC loss", "CC loss", (.08, .66))]
S = ["S1", "S2", "S3", "S4", "S5"]

def main(D, out):
    d = pd.read_csv(os.path.join(D, "f3_or_by_stratum.csv"), encoding="utf-8-sig")
    d = d[~d.event.str.contains("strict")]

    fig, axes = plt.subplots(3, 2, figsize=(W1, 3.70), sharex=True)
    fig.subplots_adjust(left=.155, right=.985, bottom=.215, top=.935,
                        hspace=.28, wspace=.14)

    x = np.arange(5)
    for r, (ev, lab, ylim) in enumerate(EV):
        for c, (key, name) in enumerate(AX.items()):
            ax = axes[r, c]
            g = d[(d.axis == key) & (d.event == ev)].set_index("stratum").reindex(S)
            if g.rate_expo.isna().all():
                
                
                ax.text(.5, .55, "n/a\n(binary answers)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=PT, color=GREY,
                        linespacing=1.2)
                ax.set_ylim(*ylim)
                ax.grid(axis="y", linestyle=":", alpha=.6)
                ax.set_axisbelow(True)
                ax.tick_params(labelsize=PT)
                ax.set_xticks(x)
                ax.set_xticklabels(S, fontsize=PT)
                continue
            ax.plot(x, g.rate_expo.values, color=EXPO, lw=1.3, marker="o",
                    ms=3.4, mfc=EXPO, zorder=3,
                    label="Visual-demanding" if r == 0 and c == 0 else None)
            ax.plot(x, g.rate_base.values, color=GREY, lw=1.3, marker="s",
                    ms=3.4, mfc="white", mew=1.0, linestyle="--", zorder=3,
                    label="Language-solvable" if r == 0 and c == 0 else None)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", linestyle=":", alpha=.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=PT)
            if c == 1:
                ax.set_yticklabels([])
            ax.set_xticks(x)
            ax.set_xticklabels(S, fontsize=PT)
        if not d[(d.axis == list(AX)[1]) & (d.event == ev)].empty:
            axes[r, 0].set_ylabel(lab, fontsize=PT)
        else:
            axes[r, 0].set_ylabel(lab, fontsize=PT)

    for c, name in enumerate(AX.values()):
        axes[0, c].set_title(name, fontsize=PT, pad=3)
    fig.text(.57, .095, "Leave-team-out difficulty quintile  (hard $\\rightarrow$ easy)",
             ha="center", fontsize=PT)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=PT, frameon=False, loc="lower center",
               bbox_to_anchor=(.57, .00), columnspacing=1.4,
               handlelength=2.0, handletextpad=.5)
    save(fig, out, "fig3_visual_axis_strata", target_w=W1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=FIGS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    main(a.data, a.out)
