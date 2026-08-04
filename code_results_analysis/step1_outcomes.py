#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

RMAX = int(os.environ.get("RLAST", "3"))   # C-03: R_last
B = 2000
SEED = 42
BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
SIT = ["MaC", "MiC", "CC", "CW", "MaW"]

def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["round"] = df["round"].astype(int)
    df["is_correct"] = df["is_correct"].astype(int)
    return df

def valid_keys(d):
    n = (d[d["round"].isin([0, RMAX])]
         .assign(ok=lambda x: (x.answer != "").astype(int))
         .groupby(["benchmark", "team_id", "question_id", "round"]).ok.sum().unstack())
    return set(n[(n[0] == 3) & (n[RMAX] == 3)].index)

def vote(answers):
    v = [a for a in answers if a]
    if not v:
        return None
    c = Counter(v)
    mx = max(c.values())
    return c, mx, {a for a, n in c.items() if n == mx}

def classify(c, mx, top, gold):
    if mx == 3:
        return "MaC" if gold in top else "MaW"
    if mx == 2:
        return "MaC" if gold in top else ("MiC" if gold in c else "MaW")
    return "CC" if gold in c else "CW"

class QBoot:

    def __init__(self, qids, B=B, seed=SEED):
        self.q, self.inv = np.unique(np.asarray(qids), return_inverse=True)
        nq = len(self.q)
        rng = np.random.default_rng(seed)
        self.C = rng.multinomial(nq, np.full(nq, 1.0 / nq), size=B).astype(np.float64)

    def ci(self, num, den):
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        N = np.bincount(self.inv, weights=num, minlength=len(self.q))
        D = np.bincount(self.inv, weights=den, minlength=len(self.q))
        bn, bd = self.C @ N, self.C @ D
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(bd > 0, bn / bd, np.nan)
        pt = N.sum() / D.sum() if D.sum() > 0 else np.nan
        return pt, float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))

def add(rows, base, name, qb, num, den):
    p, lo, hi = qb.ci(num, den)
    rows.append(dict(base, 지표=name, n=int(np.asarray(den).sum()),
                     값=p, lo=lo, hi=hi))

def build_panel(df):
    d = df[df["mode"] == "debate"]
    keys = valid_keys(d)
    A, G = {}, {}
    for r in d.itertuples(index=False):
        A[(r.benchmark, r.team_id, r.question_id, r.round, r.model)] = r.answer
        G[(r.benchmark, r.question_id)] = r.gold
    members = (d.groupby(["benchmark", "team_id"]).model
                 .apply(lambda s: tuple(sorted(set(s)))).to_dict())

    rows = []
    for (b, t, q) in keys:
        ms = members[(b, t)]
        g = G[(b, q)]
        a0 = [A.get((b, t, q, 0, m), "") for m in ms]
        a4 = [A.get((b, t, q, RMAX, m), "") for m in ms]
        c0, mx0, top0 = vote(a0)
        _, _, top4 = vote(a4)
        pool = {x for x in a0 if x}
        w = 1.0 / len(top4)
        bb = (1.0 / len(top0)) if g in top0 else 0.0
        ff = sum(w for a in top4 if a == g)
        adopt = sum(w for a in top4 if a in pool)
        rows.append(dict(
            benchmark=b, team_id=t, question_id=q,
            situation=classify(c0, mx0, top0, g),
            b=bb, f=ff, delta=ff - bb,
            채택=adopt, 창발=1.0 - adopt,
            채택_정답=sum(w for a in top4 if a in pool and a == g),
            채택_오답=sum(w for a in top4 if a in pool and a != g),
            창발_정답=sum(w for a in top4 if a not in pool and a == g),
            창발_오답=sum(w for a in top4 if a not in pool and a != g),
            n_pool=len(pool), n_top4=len(top4)))
    P = pd.DataFrame(rows)
    P["이득"] = (P.delta > 1e-12).astype(int)
    P["손실"] = (P.delta < -1e-12).astype(int)
    return P, len(keys)

def table0(P, out):
    rows = []
    for b in BM:
        G = P[P.benchmark == b]
        qb = QBoot(G.question_id)
        one = np.ones(len(G))
        for s in SIT:
            m = (G.situation == s).astype(float).values
            if m.sum() == 0:
                rows.append(dict(benchmark=b, situation=s, n=0, 비율=0.0,
                                 b평균=np.nan, f평균=np.nan, delta평균=np.nan,
                                 delta_lo=np.nan, delta_hi=np.nan))
                continue
            share, slo, shi = qb.ci(m, one)
            dp, dlo, dhi = qb.ci(G.delta.values * m, m)
            rows.append(dict(benchmark=b, situation=s, n=int(m.sum()),
                             비율=share, 비율_lo=slo, 비율_hi=shi,
                             b평균=float((G.b.values * m).sum() / m.sum()),
                             f평균=float((G.f.values * m).sum() / m.sum()),
                             delta평균=dp, delta_lo=dlo, delta_hi=dhi))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "t3f1_by_state.csv"), index=False, encoding="utf-8-sig")
    return T

def gain_loss_rows(G, base):
    qb = QBoot(G.question_id)
    one = np.ones(len(G))
    rows = []
    add(rows, base, "acc_R0", qb, G.b.values, one)
    add(rows, base, "acc_Rlast", qb, G.f.values, one)
    add(rows, base, "delta", qb, G.delta.values, one)
    add(rows, base, "이득률", qb, G.이득.values, one)
    add(rows, base, "손실률", qb, G.손실.values, one)
    for s, key, sign in [("MaC", "손실기대", -1), ("MiC", "구제기대", 1),
                         ("CC", "순변화", 1), ("CW", "구제기대", 1), ("MaW", "구제기대", 1)]:
        m = (G.situation == s).astype(float).values
        if m.sum() == 0:
            continue
        add(rows, base, f"{s}_{key}", qb, sign * G.delta.values * m, m)
    for s, key, col in [("MaC", "손실발생률", "손실"), ("MiC", "구제발생률", "이득"),
                        ("CC", "손실발생률", "손실"), ("CC", "구제발생률", "이득"),
                        ("CW", "구제발생률", "이득"), ("MaW", "구제발생률", "이득")]:
        m = (G.situation == s).astype(float).values
        if m.sum() == 0:
            continue
        add(rows, base, f"{s}_{key}", qb, G[col].values * m, m)
    return rows

def table12(P, out):
    r1 = []
    for b in BM:
        r1 += gain_loss_rows(P[P.benchmark == b], dict(benchmark=b))
    T1 = pd.DataFrame(r1).pivot(index="benchmark", columns="지표", values=["값", "lo", "hi", "n"])
    long1 = pd.DataFrame(r1)
    long1.to_csv(os.path.join(out, "t3_gain_loss.csv"), index=False, encoding="utf-8-sig")

    r2 = []
    for b in BM:
        for t, G in P[P.benchmark == b].groupby("team_id"):
            r2 += gain_loss_rows(G, dict(benchmark=b, team_id=t))
    pd.DataFrame(r2).to_csv(os.path.join(out, "extra_gain_loss_by_team.csv"),
                            index=False, encoding="utf-8-sig")
    return long1

def sec2_matrix(P, out):
    def rows_for(G, base):
        qb = QBoot(G.question_id)
        one = np.ones(len(G))
        rr = []
        for c in ["채택", "창발", "채택_정답", "채택_오답", "창발_정답", "창발_오답"]:
            add(rr, base, c, qb, G[c].values, one)
        
        add(rr, base, "정답|채택", qb, G.채택_정답.values, G.채택.values)
        add(rr, base, "정답|창발", qb, G.창발_정답.values, G.창발.values)
        return rr

    r = []
    for b in BM:
        r += rows_for(P[P.benchmark == b], dict(benchmark=b))
    pd.DataFrame(r).to_csv(os.path.join(out, "f2_adoption.csv"),
                           index=False, encoding="utf-8-sig")
    r2 = []
    for b in BM:
        for t, G in P[P.benchmark == b].groupby("team_id"):
            r2 += rows_for(G, dict(benchmark=b, team_id=t))
    pd.DataFrame(r2).to_csv(os.path.join(out, "extra_adoption_by_team.csv"),
                            index=False, encoding="utf-8-sig")

def sec2_baseline(df, P, out):
    space = (df[df.answer != ""].groupby(["benchmark", "question_id"]).answer
             .nunique().rename("K"))
    Q = P.merge(space, on=["benchmark", "question_id"], how="left")
    rows = []
    for b in BM:
        G = Q[Q.benchmark == b]
        qb = QBoot(G.question_id)
        one = np.ones(len(G))
        p, lo, hi = qb.ci(np.minimum(G.n_pool / G.K, 1.0).values, one)
        pa, la, ha = qb.ci(G.채택.values, one)
        rows.append(dict(benchmark=b, n=len(G),
                         답공간K_중앙값=float(G.K.median()),
                         R0풀크기_평균=float(G.n_pool.mean()),
                         우연채택기준선=p, 기준선_lo=lo, 기준선_hi=hi,
                         실제채택률=pa, 채택_lo=la, 채택_hi=ha,
                         초과=pa - p))
    pd.DataFrame(rows).to_csv(os.path.join(out, "f2_adoption_baseline.csv"),
                              index=False, encoding="utf-8-sig")

def sec2_selfreflect(df, P, out):
    rows = []
    sr = df[df["mode"] == "self_reflect"]
    piv = (sr[sr["round"].isin([0, RMAX])]
           .pivot_table(index=["benchmark", "question_id", "model"], columns="round",
                        values="answer", aggfunc="first"))
    piv = piv[(piv[0] != "") & (piv[RMAX] != "")]
    gold = df.drop_duplicates(["benchmark", "question_id"]).set_index(
        ["benchmark", "question_id"]).gold
    piv = piv.reset_index()
    piv["gold"] = list(gold.loc[list(zip(piv.benchmark, piv.question_id))])
    piv["유지"] = (piv[0] == piv[RMAX]).astype(float)
    piv["acc_R0"] = (piv[0] == piv.gold).astype(float)
    piv["acc_Rlast"] = (piv[RMAX] == piv.gold).astype(float)

    
    dd = df[df["mode"] == "debate"]
    dp = (dd[dd["round"].isin([0, RMAX])]
          .pivot_table(index=["benchmark", "team_id", "question_id", "model"],
                       columns="round", values="answer", aggfunc="first").reset_index())
    dp = dp[(dp[0] != "") & (dp[RMAX] != "")]
    dp["유지"] = (dp[0] == dp[RMAX]).astype(float)

    for b in BM:
        S = piv[piv.benchmark == b]
        qbs = QBoot(S.question_id)
        one = np.ones(len(S))
        for name, col in [("유지율", "유지"), ("acc_R0", "acc_R0"), ("acc_Rlast", "acc_Rlast")]:
            p, lo, hi = qbs.ci(S[col].values, one)
            rows.append(dict(benchmark=b, mode="self_reflect", 단위="에이전트",
                             지표=name, n=len(S), 값=p, lo=lo, hi=hi))
        D = dp[dp.benchmark == b]
        qbd = QBoot(D.question_id)
        p, lo, hi = qbd.ci(D.유지.values, np.ones(len(D)))
        rows.append(dict(benchmark=b, mode="debate", 단위="에이전트",
                         지표="유지율", n=len(D), 값=p, lo=lo, hi=hi))
        T = P[P.benchmark == b]
        qbt = QBoot(T.question_id)
        for name, col in [("채택률", "채택"), ("acc_R0", "b"), ("acc_Rlast", "f")]:
            p, lo, hi = qbt.ci(T[col].values, np.ones(len(T)))
            rows.append(dict(benchmark=b, mode="debate", 단위="팀",
                             지표=name, n=len(T), 값=p, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(out, "s41_selfreflect_control.csv"),
                              index=False, encoding="utf-8-sig")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.csv)
    P, nk = build_panel(df)
    print(f"[입력] {a.csv}")
    print(f"[필터] mode=debate | C-02 적용 팀-문항 {nk:,}개")
    print(f"[단위] (팀, 문항) {len(P):,}행")
    print(f"[부트] 문항 클러스터, B={B}, seed={SEED}\n")

    T0 = table0(P, a.out)
    print("[0절] R0 구성 분포")
    print("   " + T0.pivot(index="benchmark", columns="situation", values="비율")
          .reindex(BM)[[c for c in SIT]].round(4).to_string().replace("\n", "\n   "))

    T1 = table12(P, a.out)
    print("\n[1절] 벤치마크별 핵심 지표")
    piv = T1.pivot(index="benchmark", columns="지표", values="값").reindex(BM)
    cols = ["acc_R0", "acc_Rlast", "delta", "이득률", "손실률",
            "MaC_손실기대", "MiC_구제기대", "MaC_손실발생률", "MiC_구제발생률"]
    print("   " + piv[[c for c in cols if c in piv.columns]].round(4)
          .to_string().replace("\n", "\n   "))

    sec2_matrix(P, a.out)
    sec2_baseline(df, P, a.out)
    sec2_selfreflect(df, P, a.out)
    M = pd.read_csv(os.path.join(a.out, "f2_adoption.csv"), encoding="utf-8-sig")
    print("\n[2절] 창발/채택 × 정답/오답")
    print("   " + M.pivot(index="benchmark", columns="지표", values="값").reindex(BM)
          [["채택", "창발", "채택_정답", "채택_오답", "창발_정답", "창발_오답"]]
          .round(4).to_string().replace("\n", "\n   "))
    Bl = pd.read_csv(os.path.join(a.out, "f2_adoption_baseline.csv"), encoding="utf-8-sig")
    print("\n[2절] 우연 채택 기준선")
    print("   " + Bl.round(4).to_string(index=False).replace("\n", "\n   "))

    P.to_csv(os.path.join(a.out, "panel_team_question.csv.gz"),
             index=False, encoding="utf-8-sig")
    print(f"\n[저장] {a.out}/ 에 7개 CSV + panel_team_question.csv.gz")

if __name__ == "__main__":
    main()
