#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import os
import numpy as np, pandas as pd
from scipy import stats

BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
NQ = 5

def mh(df):
    a, b, c, d = df.a.values, df.b.values, df.c.values, df.d.values
    N = a + b + c + d
    ok = (N > 0) & ((a + b) > 0) & ((c + d) > 0)
    a, b, c, d, N = a[ok], b[ok], c[ok], d[ok], N[ok]
    if len(a) == 0: return np.nan
    R, S = a * d / N, b * c / N
    if R.sum() <= 0 or S.sum() <= 0: return np.nan
    return R.sum() / S.sum()

def counts(E, strata, expo):
    g = E.assign(_e=expo.astype(int), _y=E.event.astype(int))
    t = g.groupby(strata + ["_e", "_y"], observed=True).size().unstack(["_e", "_y"], fill_value=0)
    for col in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        if col not in t.columns: t[col] = 0
    return pd.DataFrame({"a": t[(1, 1)], "b": t[(1, 0)],
                         "c": t[(0, 1)], "d": t[(0, 0)]}).reset_index()

def pc(x, y, zs):
    d = pd.concat([x, y] + zs, axis=1).dropna()
    if len(d) < 5 + len(zs): return np.nan, np.nan, len(d)
    A = np.c_[np.ones(len(d)), d.iloc[:, 2:].values] if zs else np.ones((len(d), 1))
    def res(v):
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    ex, ey = res(d.iloc[:, 0].values), res(d.iloc[:, 1].values)
    if ex.std() == 0 or ey.std() == 0: return np.nan, np.nan, len(d)
    r = np.corrcoef(ex, ey)[0, 1]; n = len(d); k = A.shape[1]
    t = r * np.sqrt((n - k - 1) / max(1e-12, 1 - r ** 2))
    return r, 2 * (1 - stats.t.cdf(abs(t), n - k - 1)), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    SRC, DATA_, OUTDIR = a.csv, a.data, a.out

    RL = int(os.environ.get("RLAST", "3"))
    E = pd.read_csv(os.path.join(DATA_, "agent_events.csv"))
    E["유형"] = np.where(E.k_ok == 1, "유익", np.where(E.j_ok == 1, "유해", "무익"))

    d0 = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False,
                     usecols=["benchmark", "question_id", "mode", "round", "model", "is_correct"])
    r0 = d0[(d0["mode"] == "debate") & (d0["round"] == 0)]
    qm = r0.groupby(["benchmark", "question_id", "model"]).is_correct.mean().unstack("model")
    models = list(qm.columns)
    tm = E.groupby(["benchmark", "team_id"]).k.unique().apply(sorted)
    wi = {}
    for (b, t), mem in tm.items():
        s = qm.loc[b][[m for m in models if m not in mem]].mean(axis=1)
        for q, v in s.items(): wi[(b, t, q)] = v
    E["wi_lto"] = [wi.get((b, t, q), np.nan) for b, t, q in zip(E.benchmark, E.team_id, E.question_id)]
    E = E.dropna(subset=["wi_lto"])
    E["wi_q"] = E.groupby("benchmark").wi_lto.transform(
        lambda s: pd.qcut(s.rank(method="first"), NQ, labels=False))
    S0 = r0.groupby(["benchmark", "model"]).is_correct.mean()

    print(f"[입력] agent_events.csv 기회 {len(E):,}건 "
          f"(R1 {int((E['round']==1).sum()):,}) | R_last=R{RL}")

    
    P = pd.read_csv(os.path.join(DATA_, "panel_team_question.csv.gz"),
                    encoding="utf-8-sig")
    mic = P[P.situation == "MiC"]; mac = P[P.situation == "MaC"].assign(loss=lambda x: 1 - x.f)
    OUT = pd.DataFrame({
        "팀정확도": P.groupby(["benchmark", "team_id"]).f.mean(),
        "팀Δ": P.groupby(["benchmark", "team_id"]).delta.mean(),
        "MaC손실": mac.groupby(["benchmark", "team_id"]).loss.mean(),
        "MiC구제": mic.groupby(["benchmark", "team_id"]).f.mean()}).reset_index()
    OUT.columns = ["benchmark", "team", "팀정확도", "팀Δ", "MaC손실", "MiC구제"]
    print(f"[결과변수] panel_team_question.csv.gz  (벤치마크, 팀) {len(OUT)}셀")

    
    rows = []
    for 판본, Esub in [("R1만", E[E["round"] == 1]), ("R1-R3", E)]:
        st = ["k", "j", "consensus", "wi_q"] if 판본 == "R1만" else ["k", "j", "round", "consensus", "wi_q"]
        for b in BM:
            G = Esub[Esub.benchmark == b]
            teams = sorted(G.team_id.unique())
            for t in teams:
                sub = G[G.team_id != t]
                mem = sorted(G[G.team_id == t].k.unique())
                rec = dict(판본=판본, benchmark=b, team=t)
                
                D, BASE = [], []
                for m in mem:
                    h = sub[(sub.j == m) & (sub.유형.isin(["유익", "무익"]))]
                    if h.유형.nunique() < 2: D.append(np.nan); BASE.append(np.nan); continue
                    D.append(h[h.유형 == "유익"].event.mean() - h[h.유형 == "무익"].event.mean())
                    BASE.append(h[h.유형 == "무익"].event.mean())
                rec["위험차"] = np.nanmean(D) if np.any(~np.isnan(D)) else np.nan
                rec["기저이동"] = np.nanmean(BASE) if np.any(~np.isnan(BASE)) else np.nan
                
                for cname, (ex, ba) in [("logOR_대조1", ("유익", "무익")),
                                        ("logOR_방향성", ("유익", "유해"))]:
                    H = sub[sub.유형.isin([ex, ba])]
                    vals = []
                    for m in mem:
                        h = H[H.j == m]
                        if h.유형.nunique() < 2: vals.append(np.nan); continue
                        o = mh(counts(h, st, h.유형 == ex))
                        vals.append(np.log(o) if (o and o > 0 and np.isfinite(o)) else np.nan)
                    rec[cname] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
                s = [S0[(b, m)] for m in mem]
                rec["s_mean"] = np.mean(s)
                rec["강도격차"] = max(s) - np.mean(sorted(s)[:2])
                rows.append(rec)
    T = pd.DataFrame(rows).merge(OUT, on=["benchmark", "team"])
    VARS = ["위험차", "기저이동", "logOR_대조1", "logOR_방향성", "s_mean", "강도격차"]
    TGTS = ["MaC손실", "팀Δ", "MiC구제", "팀정확도"]
    for c in VARS + TGTS:
        T[c + "_r"] = T.groupby(["판본", "benchmark"])[c].rank()
    T.to_csv(os.path.join(OUTDIR, "t6_team_composition.csv"), index=False,
             encoding="utf-8-sig")
    print(f"[단위] (판본, 벤치마크, 팀) {len(T)}행 · 모델 지표는 leave-one-team-out\n")

    res = []
    def block(title, tgt, var, zs):
        print(f"\n  {title}")
        print(f"    {'판본':<8s}" + "".join(f"{b[:8]:>17s}" for b in BM) + "      통합")
        for 판본 in ["R1만", "R1-R3"]:
            line = f"    {판본:<8s}"
            X = T[T.판본 == 판본]
            for b in BM:
                g = X[X.benchmark == b]
                r, p, n = pc(g[var + "_r"], g[tgt + "_r"], [g[z + "_r"] for z in zs])
                line += ("n/a" if np.isnan(r) else f"{r:+.2f} ({p:.3f})").rjust(17)
                res.append(dict(판본=판본, target=tgt, var=var, ctrl="+".join(zs),
                                benchmark=b, rho=r, p=p, n=n))
            r, p, n = pc(X[var + "_r"], X[tgt + "_r"], [X[z + "_r"] for z in zs])
            line += f"   {r:+.3f} (p={p:.4f}, n={n})"
            res.append(dict(판본=판본, target=tgt, var=var, ctrl="+".join(zs),
                            benchmark="POOLED", rho=r, p=p, n=n))
            print(line)

    print("=" * 104)
    print("[A] 청자 감지력 → 팀 MaC 손실   (팀 평균 강도 통제)")
    print("=" * 104)
    block("A-1  위험차 | 강도", "MaC손실", "위험차", ["s_mean"])
    block("A-2  logOR 대조1 | 강도", "MaC손실", "logOR_대조1", ["s_mean"])
    block("A-3  logOR 방향성 | 강도", "MaC손실", "logOR_방향성", ["s_mean"])
    block("A-4  기저이동 | 강도", "MaC손실", "기저이동", ["s_mean"])

    print("\n" + "=" * 104)
    print("[B] 결정 검정 — 분별력인가 그냥 잘 흔들리는 것인가 (강도 + 상대변수 통제)")
    print("=" * 104)
    block("B-1  위험차 | 강도, 기저이동", "MaC손실", "위험차", ["s_mean", "기저이동"])
    block("B-2  기저이동 | 강도, 위험차", "MaC손실", "기저이동", ["s_mean", "위험차"])

    print("\n" + "=" * 104)
    print("[C] 토론의 이득 쪽 — 청자 감지력이 설명하는가 (강도 통제)")
    print("=" * 104)
    block("C-1  위험차 | 강도  → 팀Δ", "팀Δ", "위험차", ["s_mean"])
    block("C-2  logOR 대조1 | 강도  → 팀Δ", "팀Δ", "logOR_대조1", ["s_mean"])
    block("C-3  위험차 | 강도  → MiC 구제", "MiC구제", "위험차", ["s_mean"])
    block("C-4  logOR 대조1 | 강도  → MiC 구제", "MiC구제", "logOR_대조1", ["s_mean"])
    block("C-5  위험차 | 강도, 강도격차 → 팀Δ", "팀Δ", "위험차", ["s_mean", "강도격차"])

    print("\n" + "=" * 104)
    print("[D] 대조 — 강도 구조는 그대로인가 (3-29 재확인, 판본 무관해야 정상)")
    print("=" * 104)
    block("D-1  s_mean → 팀정확도 (무통제)", "팀정확도", "s_mean", [])
    block("D-2  강도격차 → 팀Δ (무통제)", "팀Δ", "강도격차", [])

    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUTDIR, "t6_team_composition_partial.csv"), index=False,
             encoding="utf-8-sig")
    print(f"\n[산출] {OUTDIR}/t6_team_composition.csv · {OUTDIR}/t6_team_composition_partial.csv")

if __name__ == "__main__":
    main()
