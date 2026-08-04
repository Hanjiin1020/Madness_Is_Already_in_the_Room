#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import argparse
import os

import os
RLAST = int(os.environ.get("RLAST", "3"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

for _p in ["/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
           "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
           "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf"]:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)

PT = 9          

def setup_aaai_figure_style(use_serif=True):
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    if use_serif:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman",
                                      "DejaVu Serif"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = PT
    plt.rcParams["ytick.labelsize"] = PT
    plt.rcParams["legend.fontsize"] = PT
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

setup_aaai_figure_style(use_serif=True)

W1, W2 = 3.3, 6.9          
BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
LBL = {"hallusionbench": "HallusionBench", "mmstar": "MMStar",
       "mmmu": "MMMU", "mme-cot": "MME-CoT"}
SHORT = {"hallusionbench": "HB", "mmstar": "MMStar",
         "mmmu": "MMMU", "mme-cot": "MME-CoT"}
SIT = ["MaC", "MiC", "CC", "CW", "MaW"]

DIV5 = ["#08519C", "#4292C6", "#9ECAE1", "#FDBE85", "#D94801"]
CORRECT, WRONG = "#0072B2", "#D55E00"
TINT_C, TINT_W = "#93C4E0", "#F3B183"
AGREE4 = ["#B7E4D4", "#66C2A5", "#1B9E77", "#00584A"]
RANK3 = ["#6A2C55", "#CC79A7", "#EBC0D6"]
GREY, GREYD = "#9E9E9E", "#616161"
H5 = ["", "//", "..", "xx", "\\\\"]

DARKFILL = {DIV5[0], DIV5[1], CORRECT, WRONG, AGREE4[2], AGREE4[3],
            RANK3[0], RANK3[1], GREYD}

def save(fig, out, name, target_w=None):

    from matplotlib.transforms import Bbox

    def tight():
        fig.canvas.draw()
        return fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02)

    if target_w is None:
        target_w = W2 if fig.get_size_inches()[0] > (W1 + W2) / 2 else W1
    for _ in range(3):
        bb = tight()
        r = target_w / bb.width
        if abs(r - 1.0) < 0.004:
            break
        w0, h0 = fig.get_size_inches()
        fig.set_size_inches(w0 * r, h0, forward=True)   
    bb = tight()
    if target_w > bb.width:                      
        dx = (target_w - bb.width) / 2.0
        bb = Bbox.from_extents(bb.x0 - dx, bb.y0, bb.x1 + dx, bb.y1)
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches=bb)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches=bb)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf  ({bb.width:.2f} x {bb.height:.2f} in)")

def pv(df, idx, col, val):
    return df.pivot(index=idx, columns=col, values=val).reindex(BM)

def stacked(ax, mat, cols, hats, labels, ylabels, xlabel, minlab=.10):
    y = np.arange(len(mat))[::-1]
    left = np.zeros(len(mat))
    for i, (c, h, lb) in enumerate(zip(cols, hats, labels)):
        v = np.nan_to_num(mat[:, i]) * 100
        ax.barh(y, v, left=left, height=.6, color=c, hatch=h,
                edgecolor="black", linewidth=.7, label=lb)
        for yy, ll, vv in zip(y, left, v):
            if vv >= minlab * 100:
                ax.text(ll + vv / 2, yy, f"{vv:.0f}", ha="center", va="center",
                        fontsize=PT, color="white" if c in DARKFILL else "black")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=PT)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlabel)
