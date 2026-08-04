#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import os
import argparse
from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd

RMAX = int(os.environ.get("RLAST", "3"))   # C-03: R_last
BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]

def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["round"] = df["round"].astype(int)
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)
    return df[df["mode"] == "debate"].copy()

def valid_keys(d):
    n = (d[d["round"].isin([0, RMAX])]
         .assign(ok=lambda x: (x.answer != "").astype(int))
         .groupby(["benchmark", "team_id", "question_id", "round"]).ok.sum().unstack())
    return set(n[(n[0] == 3) & (n[RMAX] == 3)].index)

def build_events(d, keys):
    A = {}
    for row in d.itertuples(index=False):
        A[(row.benchmark, row.team_id, row.question_id, row.round, row.model)] = row.answer
    gold = {(r.benchmark, r.question_id): r.gold for r in d.itertuples(index=False)}
    members = (d.groupby(["benchmark", "team_id"]).model
                 .apply(lambda s: tuple(sorted(set(s)))).to_dict())

    rows = []
    for (b, t, q) in keys:
        ms = members[(b, t)]
        g = gold[(b, q)]
        for r in range(1, RMAX + 1):
            prev = {m: A.get((b, t, q, r - 1, m), "") for m in ms}
            cur = {m: A.get((b, t, q, r, m), "") for m in ms}
            for k, j in permutations(ms, 2):
                ak, aj, cj = prev[k], prev[j], cur[j]
                if not ak or not aj or not cj:      
                    continue
                if ak == aj:                        
                    continue
                third = [m for m in ms if m not in (k, j)][0]
                rows.append((b, t, q, r, k, j,
                             int(cj == ak),          
                             int(ak == g),           
                             int(aj == g),           
                             int(prev[third] == ak)))  
    return pd.DataFrame(rows, columns=["benchmark", "team_id", "question_id", "round",
                                       "k", "j", "event", "k_ok", "j_ok", "consensus"])

def opp_type(E):
    return np.where(E.k_ok == 1, "유익",
                    np.where(E.j_ok == 1, "유해", "무익"))

def rate_table(E, by, keycol, label):
    out = []
    for (m, t), g in E.groupby([keycol, by], observed=True):
        out.append(dict(model=m, grp=t, n_기회=len(g), 비율=g.event.mean()))
    T = pd.DataFrame(out)
    piv = T.pivot(index="model", columns="grp", values="비율")
    cnt = T.pivot(index="model", columns="grp", values="n_기회")
    piv.columns = [f"{label}_{c}" for c in piv.columns]
    cnt.columns = [f"n_{c}" for c in cnt.columns]
    return piv.join(cnt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()

    d = load(a.csv)
    keys = valid_keys(d)
    print(f"[입력] {a.csv}")
    print(f"[필터] mode=debate | 팀-문항 {len(keys):,}개 (C-02 적용)")
    E = build_events(d, keys)
    E["기회유형"] = opp_type(E)
    print(f"[단위] 순서쌍 기회 {len(E):,}건\n")

    
    
    
    
    S = (d[d["round"] == 0].groupby(["benchmark", "model"]).is_correct.mean())

    for b in BM:
        G = E[E.benchmark == b]
        order = S[b].sort_values(ascending=False).index.tolist()
        print("=" * 100)
        print(f"[{b}]  기회 {len(G):,}건   (모델은 강도 내림차순)")
        print("=" * 100)

        
        ex = pd.crosstab(G.k, G.기회유형, normalize="index").reindex(order)
        exn = pd.crosstab(G.k, G.기회유형).reindex(order)
        TY = [c for c in ["유익", "유해", "무익"] if c in ex.columns]
        print("\n  [노출] 내 답이 어떤 기회에 놓이는 비율 (k 기준)")
        if "무익" not in ex.columns:
            print("      * 이진 답이라 '무익'(둘 다 오답, 답 다름) 기회가 원리적으로 없음")
        print("   " + ex[TY].round(3)
              .join(exn.sum(axis=1).rename("n")).to_string().replace("\n", "\n   "))

        
        print("\n  [(a) 상대 답 정오] 설득력 = 남이 내 답으로 오는 비율 (k 집계)")
        pk = rate_table(G, "기회유형", "k", "설득").reindex(order)
        cols = [f"설득_{c}" for c in TY] + [f"n_{c}" for c in TY]
        print("   " + pk[cols].round(4).to_string().replace("\n", "\n   "))
        pk["순영향"] = pk["설득_유익"] - pk["설득_유해"]
        print(f"\n   순영향(구제율 − 오염율): " +
              "  ".join(f"{m} {pk.loc[m,'순영향']:+.4f}" for m in order))

        print("\n  [(a) 상대 답 정오] 피설득성 = 내가 남의 답으로 가는 비율 (j 집계)")
        pj = rate_table(G, "기회유형", "j", "피설득").reindex(order)
        print("   " + pj[[f"피설득_{c}" for c in TY]].round(4).to_string().replace("\n", "\n   "))

        
        print("\n  [(b) 내 답 정오] 피설득성 — 내 답이 정답/오답일 때 떠나는 비율 (j 집계)")
        G2 = G.assign(내답=np.where(G.j_ok == 1, "내답정답", "내답오답"))
        pj2 = rate_table(G2, "내답", "j", "피설득").reindex(order)
        print("   " + pj2.round(4).to_string().replace("\n", "\n   "))
        print("      * 내답정답 = 유해 기회와 동일 / 내답오답 = 유익 + 무익")

        print("\n  [(b) 내 답 정오] 설득력 — 상대 답이 정답/오답일 때 끌어오는 비율 (k 집계)")
        G3 = G.assign(상대답=np.where(G.j_ok == 1, "상대정답", "상대오답"))
        pk2 = rate_table(G3, "상대답", "k", "설득").reindex(order)
        print("   " + pk2.round(4).to_string().replace("\n", "\n   "))

        # (c) 2x2
        print("\n  [(c) 2×2] k 정오 × j 정오  (사건률 / 기회수)")
        cc = G.assign(cell=G.k_ok.astype(str) + G.j_ok.astype(str))
        tab = cc.groupby("cell").agg(기회=("event", "size"), 사건률=("event", "mean"))
        tab.index = tab.index.map({"10": "k정답·j오답 (유익)", "01": "k오답·j정답 (유해)",
                                   "00": "둘 다 오답 (무익)", "11": "둘 다 정답 (불가능)"})
        print("   " + tab.round(4).to_string().replace("\n", "\n   "))

        
        print("\n  [단독 vs 합의] 기회유형 × 나머지 peer 가 같은 답을 갖는가")
        sc = G.groupby(["기회유형", "consensus"]).agg(기회=("event", "size"),
                                                     사건률=("event", "mean")).reset_index()
        sc["consensus"] = sc.consensus.map({0: "단독", 1: "합의"})
        print("   " + sc.pivot(index="기회유형", columns="consensus",
                               values=["기회", "사건률"]).round(4)
              .to_string().replace("\n", "\n   "))

        
        print("\n  [라운드별] 기회유형별 사건률")
        rr = G.pivot_table(index="기회유형", columns="round", values="event", aggfunc="mean")
        print("   " + rr.round(4).to_string().replace("\n", "\n   "))
        print()

    E.to_csv(os.path.join(a.out, "agent_events.csv"), index=False)

    
    rows = []
    for b in BM:
        G = E[E.benchmark == b]
        r = G[G.기회유형 == "유익"].groupby("k").event.agg(["mean", "sum"])
        c = G[G.기회유형 == "유해"].groupby("k").event.agg(["mean", "sum"])
        M = r.join(c, lsuffix="_r", rsuffix="_c").fillna(0)
        for m in M.index:
            rows.append(dict(benchmark=b, model=m, 강도=float(S[(b, m)]),
                             구제율=float(M.loc[m, "mean_r"]),
                             오염율=float(M.loc[m, "mean_c"]),
                             비율순영향=float(M.loc[m, "mean_r"] - M.loc[m, "mean_c"]),
                             구제건수=int(M.loc[m, "sum_r"]),
                             오염건수=int(M.loc[m, "sum_c"]),
                             절대순영향=int(M.loc[m, "sum_r"] - M.loc[m, "sum_c"])))
    N = pd.DataFrame(rows)
    N["비율순위"] = N.groupby("benchmark").비율순영향.rank(ascending=False).astype(int)
    N["절대순위"] = N.groupby("benchmark").절대순영향.rank(ascending=False).astype(int)
    N.to_csv(os.path.join(a.out, "extra_net_influence.csv"), index=False,
         encoding="utf-8-sig")
    print(f"[저장] {a.out}/agent_events.csv, {a.out}/extra_net_influence.csv")

if __name__ == "__main__":
    main()
