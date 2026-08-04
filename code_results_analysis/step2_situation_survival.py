#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import RAW, DATA, FIGS  # noqa: E402
import argparse
import os
from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd

RMAX = int(os.environ.get("RLAST", "3"))   # C-03: R_last
B = 2000
SEED = 42
BM = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
NEAR = 0.005          

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

# ── 3-1 ───────────────────────────────────────────────────────────────────
def build_events(d, keys, rankof):
    A = {}
    for r in d.itertuples(index=False):
        A[(r.benchmark, r.team_id, r.question_id, r.round, r.model)] = r.answer
    members = (d.groupby(["benchmark", "team_id"]).model
                 .apply(lambda s: tuple(sorted(set(s)))).to_dict())
    rows = []
    for (b, t, q) in keys:
        ms = members[(b, t)]
        for r in range(1, RMAX + 1):
            prev = {m: A.get((b, t, q, r - 1, m), "") for m in ms}
            cur = {m: A.get((b, t, q, r, m), "") for m in ms}
            if any(not prev[m] for m in ms):
                continue                       
            for k in ms:
                if not cur[k]:
                    continue
                a = prev[k]
                peers = [m for m in ms if m != k]
                pa, pb = prev[peers[0]], prev[peers[1]]
                same = [m for m in peers if prev[m] == a]
                if pa == pb:
                    sit = "Uni" if pa == a else "Iso"
                elif len(same) == 1:
                    sit = "Sup"
                else:
                    sit = "Spl"
                c = cur[k]
                if c == a:
                    resp, src = "유지", ""
                elif c in (pa, pb):
                    resp = "수용"
                    srcm = [m for m in peers if prev[m] == c]
                    src = f"rank{min(rankof[(b, t, m)] for m in srcm)}"
                else:
                    resp, src = "신규", ""
                rows.append((b, t, q, r, k, rankof[(b, t, k)], sit, resp, src))
    return pd.DataFrame(rows, columns=["benchmark", "team_id", "question_id", "round",
                                       "model", "rank", "상황", "반응", "수용출처"])

def sec3_1(E, out):
    def tab(E, by, fname):
        rows = []
        for b in BM:
            G = E[E.benchmark == b]
            qb = QBoot(G.question_id)
            one = np.ones(len(G))
            for keys_, g in G.groupby(by, observed=True):
                keys_ = keys_ if isinstance(keys_, tuple) else (keys_,)
                base = dict(benchmark=b, **dict(zip(by if isinstance(by, list) else [by], keys_)))
                mask = np.zeros(len(G))
                mask[G.index.get_indexer(g.index)] = 1.0
                expo, elo, ehi = qb.ci(mask, one)
                row = dict(base, n=int(mask.sum()), 노출=expo, 노출_lo=elo, 노출_hi=ehi)
                for resp in ["유지", "수용", "신규"]:
                    num = ((G.반응 == resp).values * mask)
                    p, lo, hi = qb.ci(num, mask)
                    row[resp] = p
                    row[resp + "_lo"] = lo
                    row[resp + "_hi"] = hi
                rows.append(row)
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(out, fname), index=False, encoding="utf-8-sig")
        return T

    E = E.reset_index(drop=True)
    T = tab(E, ["상황"], "t4_situation_response.csv")
    tab(E, ["round", "상황"], "extra_situation_by_round.csv")
    tab(E, ["model", "상황"], "extra_situation_by_model.csv")

    
    rows = []
    Bsit = E[(E.상황 == "Spl") & (E.반응 == "수용")]
    for b in BM:
        G = E[(E.benchmark == b) & (E.상황 == "Spl")].reset_index(drop=True)
        if not len(G):
            continue
        qb = QBoot(G.question_id)
        m = (G.반응 == "수용").values.astype(float)
        for rk in ["rank1", "rank2", "rank3"]:
            num = ((G.수용출처 == rk).values.astype(float))
            p, lo, hi = qb.ci(num, m)
            rows.append(dict(benchmark=b, 출처=rk, n=int(num.sum()),
                             수용중비율=p, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(out, "extra_spl_adoption_source.csv"),
                              index=False, encoding="utf-8-sig")
    return T

# ── 3-2 ───────────────────────────────────────────────────────────────────
def sec3_2(d, keys, rankof, strength, out):
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
        a0 = {m: A.get((b, t, q, 0, m), "") for m in ms}
        holders = [m for m in ms if a0[m] == g]
        if len(holders) != 1:
            continue
        c0, mx0, top0 = vote(list(a0.values()))
        sit = classify(c0, mx0, top0, g)
        _, _, top4 = vote([A.get((b, t, q, RMAX, m), "") for m in ms])
        f = sum(1.0 / len(top4) for a in top4 if a == g)
        h = holders[0]
        rows.append(dict(benchmark=b, team_id=t, question_id=q, situation=sit,
                         holder=h, rank=rankof[(b, t, h)], f=f,
                         생존=int(f > 0.5)))
    S = pd.DataFrame(rows)

    def make(S, fname, note):
        rows = []
        for b in BM:
            X = S[S.benchmark == b]
            if not len(X):
                continue
            qb = QBoot(X.question_id)
            for sit in ["MiC", "CC"]:
                for rk in [1, 2, 3]:
                    m = ((X.situation == sit) & (X["rank"] == rk)).values.astype(float)
                    if m.sum() == 0:
                        continue
                    p, lo, hi = qb.ci(X.f.values * m, m)
                    e, elo, ehi = qb.ci(X.생존.values * m, m)
                    rows.append(dict(benchmark=b, situation=sit, rank=rk, n=int(m.sum()),
                                     기준선=0.0 if sit == "MiC" else 1 / 3,
                                     생존율_기대=p, lo=lo, hi=hi,
                                     생존율_이벤트=e, e_lo=elo, e_hi=ehi, 비고=note))
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(out, fname), index=False, encoding="utf-8-sig")
        return T

    T = make(S, "t5_survival.csv", "전체")

    
    drop = set()
    for b in BM:
        s = strength[b]
        for t in S[S.benchmark == b].team_id.unique():
            ms = sorted(members[(b, t)], key=lambda m: -s[m])
            if s[ms[0]] - s[ms[1]] < NEAR:
                drop.add((b, t))
    S2 = S[[(b, t) not in drop for b, t in zip(S.benchmark, S.team_id)]]
    make(S2, "t5_survival_sensitivity.csv",
         f"1·2위 강도격차<{NEAR} 팀 제외 ({len(drop)}팀)")
    print(f"  [3-2] 근접 순위로 제외한 (벤치,팀) = {sorted(drop)}")
    return T

def sec5(df, keys, out):
    d = df[df["mode"] == "debate"]
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
        for r in range(0, RMAX + 1):
            v = vote([A.get((b, t, q, r, m), "") for m in ms])
            if v is None:
                continue
            _, _, top = v
            rows.append((b, t, q, r, sum(1.0 / len(top) for a in top if a == g)))
    T = pd.DataFrame(rows, columns=["benchmark", "team_id", "question_id", "round", "acc"])

    out_rows = []
    for b in BM:
        X = T[T.benchmark == b]
        qb = QBoot(X.question_id)
        for r in range(0, RMAX + 1):
            m = (X["round"] == r).values.astype(float)
            p, lo, hi = qb.ci(X.acc.values * m, m)
            out_rows.append(dict(benchmark=b, 계열="debate_팀다수결", round=r,
                                 n=int(m.sum()), acc=p, lo=lo, hi=hi))
    
    okq = {}
    for (b, t, q) in keys:
        okq.setdefault(b, set()).add(q)
    # blind(회색 이미지) 조건은 논문에서 쓰지 않아 배포 자료에 없다. 데이터에 있으면 함께 낸다.
    series = [("debate", "debate_에이전트평균", range(0, RMAX + 1)),
              ("self_reflect", "self_reflect", range(0, RMAX + 1)),
              ("blind", "blind", [0])]
    have = set(df["mode"].unique())
    for mode, name, rounds in [x for x in series if x[0] in have]:
        M = df[(df["mode"] == mode) & (df.answer != "")]
        if not len(M):                       # 빈 선택에 리스트 마스크를 쓰면
            continue                         # 열 선택으로 해석되어 조용히 깨진다
        if mode == "debate":
            M = M[[(b, t, q) in keys for b, t, q in
                   zip(M.benchmark, M.team_id, M.question_id)]]
        else:
            M = M[[q in okq.get(b, set()) for b, q in zip(M.benchmark, M.question_id)]]
        for b in BM:
            X = M[M.benchmark == b]
            if not len(X):
                continue
            qb = QBoot(X.question_id)
            for r in rounds:
                m = (X["round"] == r).values.astype(float)
                if m.sum() == 0:
                    continue
                p, lo, hi = qb.ci(X.is_correct.values * m, m)
                out_rows.append(dict(benchmark=b, 계열=name, round=r,
                                     n=int(m.sum()), acc=p, lo=lo, hi=hi))
    
    
    at = df.drop_duplicates(["benchmark", "question_id"])[
        ["benchmark", "question_id", "answer_type"]]
    mcq = set(at[(at.benchmark == "mme-cot") & (at.answer_type != "freeform")].question_id)
    Xm = T[(T.benchmark == "mme-cot") & (T.question_id.isin(mcq))]
    if len(Xm):
        qb = QBoot(Xm.question_id)
        for r in range(0, RMAX + 1):
            m = (Xm["round"] == r).values.astype(float)
            p, lo, hi = qb.ci(Xm.acc.values * m, m)
            out_rows.append(dict(benchmark="mme-cot", 계열="debate_팀다수결_mcq만",
                                 round=r, n=int(m.sum()), acc=p, lo=lo, hi=hi))
        Am = df[(df["mode"] == "debate") & (df.benchmark == "mme-cot")
                & (df.answer != "") & (df.question_id.isin(mcq))]
        Am = Am[[(b, t, q) in keys for b, t, q in
                 zip(Am.benchmark, Am.team_id, Am.question_id)]]
        qb2 = QBoot(Am.question_id)
        for r in range(0, RMAX + 1):
            m = (Am["round"] == r).values.astype(float)
            p, lo, hi = qb2.ci(Am.is_correct.values * m, m)
            out_rows.append(dict(benchmark="mme-cot", 계열="debate_에이전트평균_mcq만",
                                 round=r, n=int(m.sum()), acc=p, lo=lo, hi=hi))

    R = pd.DataFrame(out_rows)
    R.to_csv(os.path.join(out, "s46_round_trend.csv"), index=False, encoding="utf-8-sig")
    (T.groupby(["benchmark", "team_id", "round"]).acc.mean().rename("acc").reset_index()
     .to_csv(os.path.join(out, "s46_round_trend_by_team.csv"),
             index=False, encoding="utf-8-sig"))
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.csv)
    d = df[df["mode"] == "debate"]
    keys = valid_keys(d)
    strength = {b: g.groupby("model").is_correct.mean()
                for b, g in d[d["round"] == 0].groupby("benchmark")}
    members = (d.groupby(["benchmark", "team_id"]).model
                 .apply(lambda s: tuple(sorted(set(s)))).to_dict())
    rankof = {}
    for (b, t), ms in members.items():
        for i, m in enumerate(sorted(ms, key=lambda x: -strength[b][x]), 1):
            rankof[(b, t, m)] = i

    print(f"[입력] {a.csv}")
    print(f"[필터] C-02 팀-문항 {len(keys):,}개 | 부트 B={B}, seed={SEED}\n")

    E = build_events(d, keys, rankof)
    print(f"[3-1] 기회 {len(E):,}건 (팀·문항·라운드·에이전트)")
    T = sec3_1(E, a.out)
    print("   " + T.pivot(index="benchmark", columns="상황",
                          values="노출").reindex(BM).round(4)
          .to_string().replace("\n", "\n   "))
    print("   반응률(수용):")
    print("   " + T.pivot(index="benchmark", columns="상황",
                          values="수용").reindex(BM).round(4)
          .to_string().replace("\n", "\n   "))

    print("\n[3-2] 순위별 단독 정답 생존율")
    S = sec3_2(d, keys, rankof, strength, a.out)
    print("   " + S.pivot_table(index=["benchmark", "situation"], columns="rank",
                                values="생존율_기대").round(4)
          .to_string().replace("\n", "\n   "))

    print("\n[5절] 라운드 추이")
    R = sec5(df, keys, a.out)
    print("   " + R[R.계열 == "debate_팀다수결"].pivot(index="benchmark", columns="round",
                                                   values="acc").reindex(BM).round(4)
          .to_string().replace("\n", "\n   "))
    print(f"\n[저장] {a.out}/ 에 7개 CSV")

if __name__ == "__main__":
    main()
