#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA  # noqa: E402

import argparse  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

RMAX = int(os.environ.get("RLAST", "3"))
BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
SEED = 42

def vote(a):
    v = [x for x in a if x]
    if len(v) < 3:
        return None
    c = Counter(v)
    mx = max(c.values())
    return c, mx, {k for k, n in c.items() if n == mx}

def classify(c, mx, top, g):
    if mx == 3:
        return "MaC" if g in top else "MaW"
    if mx == 2:
        return "MaC" if g in top else ("MiC" if g in c else "MaW")
    return "CC" if g in c else "CW"

def build(df):
    d = df[df["mode"] == "debate"]
    
    wi = (d[d["round"] == 0].groupby(["benchmark", "question_id", "model"])
          .is_correct.mean())

    rows = []
    for (bm, t, q), g in d.groupby(["benchmark", "team_id", "question_id"]):
        gold = g.gold.iloc[0]
        g0 = g[g["round"] == 0]
        r0, rl = vote(list(g0.answer)), vote(list(g[g["round"] == RMAX].answer))
        if r0 is None or rl is None:
            continue
        c0, mx0, t0 = r0
        _, _, tl = rl
        sit = classify(c0, mx0, t0, gold)
        if sit not in ("MiC", "CC"):
            continue
        holders = g0[g0.answer == gold]
        if len(holders) != 1:            
            continue
        members = sorted(g.model.unique())
        wq = wi.loc[bm, q]
        out = [m for m in wq.index if m not in members]
        rows.append(dict(
            benchmark=bm, team_id=t, question_id=q, situation=sit,
            holder=holders.model.iloc[0],
            f=(1 / len(tl)) if gold in tl else 0.0,
            wi_lto=float(wq[out].mean()) if out else np.nan,
            category=g.category.iloc[0],
            members="|".join(members)))
    return pd.DataFrame(rows)

def add_rank(E, d):
    acc = (d[(d["mode"] == "debate") & (d["round"] == 0)]
           .groupby(["benchmark", "model"]).is_correct.mean())
    r = []
    for _, row in E.iterrows():
        ms = row.members.split("|")
        order = sorted(ms, key=lambda m: -acc.loc[row.benchmark, m])
        r.append(order.index(row.holder) + 1)
    E = E.copy()
    E["rank"] = r
    return E

def frac_logit(E, bm):

    g = E[E.benchmark == bm].copy()
    if len(g) < 50 or g["rank"].nunique() < 3:
        return []
    g["q5"] = pd.qcut(g.wi_lto.rank(method="first"), 5,
                      labels=[f"S{i+1}" for i in range(5)])
    cols = ["rank", "q5", "situation"]
    if bm == "hallusionbench":
        g["visual"] = g.category.astype(str).str[:2]      # VD / VS
        cols.append("visual")
    X = pd.get_dummies(g[cols].astype({"rank": str}),
                       drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add")
    m = sm.GLM(g.f.values, X.values, family=sm.families.Binomial())
    try:
        res = m.fit(cov_type="cluster",
                    cov_kwds={"groups": pd.factorize(g.question_id)[0]})
    except Exception as e:                      
        print(f"    [{bm}] 회귀 실패: {e}")
        return []
    out = []
    for i, name in enumerate(X.columns):
        if not str(name).startswith("rank_"):
            continue
        out.append(dict(benchmark=bm, term=str(name), n=len(g),
                        coef=res.params[i], se=res.bse[i],
                        OR=float(np.exp(res.params[i])),
                        OR_lo=float(np.exp(res.conf_int()[i][0])),
                        OR_hi=float(np.exp(res.conf_int()[i][1])),
                        p=res.pvalues[i]))
    return out

def crossfit(E, d):
    d0 = d[(d["mode"] == "debate") & (d["round"] == 0)]
    rng = np.random.default_rng(SEED)
    rows = []
    for bm in BM:
        g = E[E.benchmark == bm]
        if not len(g):
            continue
        qs = d0[d0.benchmark == bm].question_id.unique()
        half = rng.permutation(len(qs)) < len(qs) // 2
        A = set(qs[half])
        for fold, fit_qs in [("A→B", A), ("B→A", set(qs) - A)]:
            acc = (d0[(d0.benchmark == bm) & (d0.question_id.isin(fit_qs))]
                   .groupby("model").is_correct.mean())
            ev = g[~g.question_id.isin(fit_qs)]
            for _, row in ev.iterrows():
                ms = row.members.split("|")
                order = sorted(ms, key=lambda m: -acc.loc[m])
                rows.append(dict(benchmark=bm, fold=fold, situation=row.situation,
                                 rank=order.index(row.holder) + 1, f=row.f))
    X = pd.DataFrame(rows)
    return (X.groupby(["benchmark", "situation", "rank"])
              .agg(n=("f", "size"), survival=("f", "mean")).reset_index())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--out", default=DATA)
    ap.add_argument("--rebuild", action="store_true",
                    help="보유자 이벤트 표를 캐시 무시하고 다시 만든다")
    a = ap.parse_args()

    d = pd.read_csv(a.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    d["round"] = d["round"].astype(int)
    d["is_correct"] = d["is_correct"].astype(int)
    print(f"[입력] {a.csv}")

    cache = os.path.join(a.out, "supB_holder_events.csv")
    E = None
    if os.path.exists(cache) and not a.rebuild:
        E = pd.read_csv(cache)
        if E.get("rlast", pd.Series([None])).iloc[0] != RMAX:
            print(f"[캐시 무시] {cache} 는 R_last=R{E.get('rlast', pd.Series(['?'])).iloc[0]} "
                  f"로 만들어졌고 지금은 R{RMAX} 다 — 다시 만든다")
            E = None
        else:
            print(f"[캐시] {cache}")
    if E is None:
        E = add_rank(build(d), d)
        E["rlast"] = RMAX
        E.to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[단위] 단독 정답 보유자가 있는 (팀,문항) {len(E):,}건 "
          f"— {dict(E.situation.value_counts())}\n")

    print("[대조] 통제 없는 순위별 생존율 (본문 Table 5)")
    print(E.groupby(["benchmark", "situation", "rank"]).f.mean()
          .unstack().round(3).to_string())

    print("\n[B-1] 통제 회귀 — 생존 ~ 순위 + 난이도5분위 + 상태 (+HB 시각라벨)")
    print("      기준 = rank 1. 모델·팀 고정효과는 순위와 공선이라 제외했다.")
    rows = []
    for bm in BM:
        rows += frac_logit(E, bm)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "supB_rank_regression.csv"),
             index=False, encoding="utf-8-sig")
    print(R.round(4).to_string(index=False) if len(R) else "  (없음)")

    print("\n[B-2] 교차적합 순위별 생존율 — 순위를 정한 문항과 평가 문항이 겹치지 않음")
    C = crossfit(E, d)
    C.to_csv(os.path.join(a.out, "supB_rank_crossfit.csv"),
             index=False, encoding="utf-8-sig")
    print(C.pivot_table(index=["benchmark", "situation"], columns="rank",
                        values="survival").round(3).to_string())
    print(f"\n[저장] {a.out}/supB_rank_regression.csv, {a.out}/supB_rank_crossfit.csv")

if __name__ == "__main__":
    main()
