#!/usr/bin/env python3
"""Build the Section 1.2, 1.6, and 1.7 robustness tables."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from math import erfc, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, RAW  # noqa: E402
from _stats import mh_or  # noqa: E402

BENCHES = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
LABELS = {
    "hallusionbench": "HallusionBench",
    "mmstar": "MMStar",
    "mmmu": "MMMU",
    "mme-cot": "MME-CoT",
}

def majority(answers: list[str], gold: str) -> dict[str, object]:
    valid = [answer for answer in answers if answer]
    if not valid:
        return {"top": set(), "tie": True, "expected": np.nan,
                "wrong": np.nan, "correct": np.nan, "oracle": 0.0}
    counts = Counter(valid)
    peak = max(counts.values())
    top = {answer for answer, count in counts.items() if count == peak}
    hit = gold in top
    return {
        "top": top,
        "tie": len(top) > 1,
        "expected": float(hit) / len(top),
        "wrong": 0.0 if len(top) > 1 else float(hit),
        "correct": float(hit),
        "oracle": float(gold in valid),
    }

def cmh_p(tables: list[tuple[float, float, float, float]]) -> float:
    tables = [table for table in tables if sum(table) > 1]
    expected = sum((a + b) * (a + c) / sum(table)
                   for table in tables for a, b, c, d in [table])
    observed = sum(a for a, _, _, _ in tables)
    variance = sum(
        (a + b) * (c + d) * (a + c) * (b + d)
        / (sum(table) ** 2 * (sum(table) - 1))
        for table in tables for a, b, c, d in [table]
    )
    return erfc(sqrt(((observed - expected) ** 2 / variance) / 2))

def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return (a * d) / (b * c) if b and c else np.nan

def visual_panel(raw: pd.DataFrame, final_round: int) -> pd.DataFrame:
    hb = raw[(raw.benchmark == "hallusionbench")
             & (raw["mode"] == "debate")
             & raw["round"].isin([0, final_round])].copy()
    r0 = hb[hb["round"] == 0]
    difficulty = r0.groupby(["question_id", "model"]).is_correct.mean().unstack()
    difficulty = difficulty.mean(axis=1)
    records = []
    for (team, question), trajectory in hb.groupby(["team_id", "question_id"]):
        gold = trajectory.gold.iloc[0]
        by_round = {round_: group for round_, group in trajectory.groupby("round")}
        if 0 not in by_round or final_round not in by_round:
            continue
        start = majority(by_round[0].answer.tolist(), gold)["expected"]
        final = majority(by_round[final_round].answer.tolist(), gold)["expected"]
        records.append({
            "team_id": team,
            "question_id": question,
            "cat": trajectory.category.iloc[0],
            "wi_acc": float(difficulty.loc[question]),
            "start_correct": bool(start > 0.5),
            "final_correct": bool(final > 0.5),
        })
    return pd.DataFrame(records)

def visual_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["damage", "recovery_failure"]:
        if outcome == "damage":
            eligible = panel[panel.start_correct].copy()
            eligible["event"] = ~eligible.final_correct
        else:
            eligible = panel[~panel.start_correct].copy()
            eligible["event"] = ~eligible.final_correct

        cells = {}
        for cat in ["VD", "VS"]:
            group = eligible[eligible.cat == cat]
            events = int(group.event.sum())
            cells[cat] = (events, len(group) - events)

        tables = []
        eligible["stratum"] = eligible.wi_acc.round(12)
        for _, group in eligible.groupby("stratum"):
            table = []
            for cat in ["VD", "VS"]:
                part = group[group.cat == cat]
                events = int(part.event.sum())
                table.extend([events, len(part) - events])
            tables.append(tuple(table))

        mh, lo, hi = mh_or(tables)
        a, b = cells["VD"]
        c, d = cells["VS"]
        rows.append({
            "outcome": outcome,
            "vd_events": a,
            "vd_n": a + b,
            "vd_rate": a / (a + b),
            "vs_events": c,
            "vs_n": c + d,
            "vs_rate": c / (c + d),
            "crude_or": odds_ratio(a, b, c, d),
            "mh_or": mh,
            "lo": lo,
            "hi": hi,
            "cmh_p": cmh_p(tables),
        })
    return pd.DataFrame(rows)

def weighted_choice(group: pd.DataFrame,
                    weights: dict[tuple[str, str, str], float], gold: str) -> float:
    scores: dict[str, float] = {}
    for row in group.itertuples(index=False):
        if row.answer:
            key = (row.benchmark, row.question_id, row.model)
            scores[row.answer] = scores.get(row.answer, 0.0) + weights[key]
    if not scores:
        return np.nan
    best = max(scores.values())
    top = [answer for answer, score in scores.items() if np.isclose(score, best)]
    return float(gold in top) / len(top)

def rank_tie_choice(group: pd.DataFrame, top_answers: set[str],
                    weights: dict[tuple[str, str, str], float], gold: str) -> float:
    candidates = []
    for answer in top_answers:
        holders = group[group.answer == answer]
        score = max(weights[(row.benchmark, row.question_id, row.model)]
                    for row in holders.itertuples(index=False))
        candidates.append((answer, score))
    best = max(score for _, score in candidates)
    selected = [answer for answer, score in candidates if np.isclose(score, best)]
    return float(gold in selected) / len(selected)

def cluster_bootstrap(frame: pd.DataFrame, columns: list[str], boot: int,
                      seed: int) -> tuple[dict[str, tuple[float, float, float]], np.ndarray]:
    grouped_sum = frame.groupby("question_id")[columns].sum(min_count=1)
    grouped_n = frame.groupby("question_id")[columns].count()
    sums = grouped_sum.to_numpy(float)
    counts = grouped_n.to_numpy(float)
    point = np.nansum(sums, axis=0) / np.nansum(counts, axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((boot, len(columns)), dtype=float)
    for start in range(0, boot, 500):
        size = min(500, boot - start)
        indices = rng.integers(0, len(grouped_sum), size=(size, len(grouped_sum)))
        draws[start:start + size] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    result = {
        column: (float(point[j]), float(np.percentile(draws[:, j], 2.5)),
                 float(np.percentile(draws[:, j], 97.5)))
        for j, column in enumerate(columns)
    }
    return result, draws

def audit_panel(raw: pd.DataFrame, endpoint: int) -> pd.DataFrame:
    debate_all = raw[raw["mode"] == "debate"].copy()
    debate = debate_all[debate_all["round"].isin([0, 1, endpoint])].copy()
    keys = ["benchmark", "team_id", "question_id"]
    endpoints = debate[debate["round"].isin([0, endpoint])].copy()
    endpoints["valid"] = endpoints.answer.str.strip().ne("")
    counts = endpoints.groupby(keys + ["round"]).valid.sum().unstack("round")
    keep_index = counts[(counts.get(0, 0) == 3) & (counts.get(endpoint, 0) == 3)].index
    keep = pd.DataFrame(list(keep_index), columns=keys)
    debate = debate.merge(keep, on=keys, how="inner", validate="many_to_one")

    r0 = debate[debate["round"] == 0]
    totals = r0.groupby(["benchmark", "model"]).is_correct.agg(["sum", "count"])
    per_question = r0.groupby(["benchmark", "model", "question_id"]).is_correct.agg(["sum", "count"])
    weights = {}
    for (bench, model, question), row in per_question.iterrows():
        total = totals.loc[(bench, model)]
        weights[(bench, question, model)] = float(
            (total["sum"] - row["sum"]) / (total["count"] - row["count"])
        )

    all_r0 = debate_all[debate_all["round"] == 0]
    model_accuracy = all_r0.groupby(["benchmark", "model"]).is_correct.mean()
    best_models = {bench: model_accuracy.loc[bench].idxmax() for bench in BENCHES}
    best_single = {}
    sc6 = {}
    for bench, model in best_models.items():
        subset = all_r0[(all_r0.benchmark == bench) & (all_r0.model == model)]
        for question, group in subset.groupby("question_id"):
            best_single[(bench, question)] = float(group.is_correct.mean())
            sc6[(bench, question)] = majority(group.answer.tolist(), group.gold.iloc[0])["expected"]

    records = []
    for (bench, team, question), trajectory in debate.groupby(keys, sort=False):
        by_round = {round_: group for round_, group in trajectory.groupby("round")}
        gold = trajectory.gold.iloc[0]
        info = {round_: majority(by_round[round_].answer.tolist(), gold)
                for round_ in [0, 1, endpoint]}
        r0_group = by_round[0]
        final_group = by_round[endpoint]
        records.append({
            "benchmark": bench,
            "team_id": team,
            "question_id": question,
            "best_single": best_single[(bench, question)],
            "r0_majority": info[0]["expected"],
            "rank_weighted": weighted_choice(r0_group, weights, gold),
            "sc6_global_best": sc6[(bench, question)],
            "r1_majority": info[1]["expected"],
            "r3_majority": info[endpoint]["expected"],
            "r0_oracle": info[0]["oracle"],
            "r0_tie": bool(info[0]["tie"]),
            "r3_tie": bool(info[endpoint]["tie"]),
            "r0_expected_random": info[0]["expected"],
            "r3_expected_random": info[endpoint]["expected"],
            "r0_tie_as_wrong": info[0]["wrong"],
            "r3_tie_as_wrong": info[endpoint]["wrong"],
            "r0_tie_as_correct": info[0]["correct"],
            "r3_tie_as_correct": info[endpoint]["correct"],
            "r0_rank_based": rank_tie_choice(r0_group, info[0]["top"], weights, gold),
            "r3_rank_based": rank_tie_choice(final_group, info[endpoint]["top"], weights, gold),
        })
    return pd.DataFrame(records)

def baseline_summary(panel: pd.DataFrame, boot: int, seed: int) -> pd.DataFrame:
    methods = ["best_single", "r0_majority", "rank_weighted", "sc6_global_best",
               "r1_majority", "r3_majority", "r0_oracle"]
    rows = []
    for i, bench in enumerate(BENCHES):
        frame = panel[panel.benchmark == bench]
        estimates, draws = cluster_bootstrap(frame, methods, boot, seed + i)
        r3_index = methods.index("r3_majority")
        for j, method in enumerate(methods):
            difference = draws[:, j] - draws[:, r3_index]
            point, lo, hi = estimates[method]
            rows.append({
                "benchmark": bench,
                "method": method,
                "accuracy": point,
                "lo": lo,
                "hi": hi,
                "diff_vs_r3": float(difference.mean()),
                "diff_lo": float(np.percentile(difference, 2.5)),
                "diff_hi": float(np.percentile(difference, 97.5)),
                "n_team_question": len(frame),
            })
    return pd.DataFrame(rows)

def tie_summary(panel: pd.DataFrame, boot: int, seed: int) -> pd.DataFrame:
    rules = {
        "expected_random": ("r0_expected_random", "r3_expected_random"),
        "tie_as_wrong": ("r0_tie_as_wrong", "r3_tie_as_wrong"),
        "tie_as_correct": ("r0_tie_as_correct", "r3_tie_as_correct"),
        "rank_based": ("r0_rank_based", "r3_rank_based"),
    }
    rows = []
    for i, bench in enumerate(BENCHES):
        frame = panel[panel.benchmark == bench].copy()
        for rule, columns in rules.items():
            estimates, draws = cluster_bootstrap(frame, list(columns), boot, seed + 20 + i)
            delta = draws[:, 1] - draws[:, 0]
            rows.append({
                "benchmark": bench,
                "rule": rule,
                "acc_r0": estimates[columns[0]][0],
                "acc_r3": estimates[columns[1]][0],
                "delta": estimates[columns[1]][0] - estimates[columns[0]][0],
                "lo": float(np.percentile(delta, 2.5)),
                "hi": float(np.percentile(delta, 97.5)),
                "tie_rate_r0": float(frame.r0_tie.mean()),
                "tie_rate_r3": float(frame.r3_tie.mean()),
                "n_team_question": len(frame),
            })
        tie_free = frame[~frame.r0_tie & ~frame.r3_tie]
        columns = ["r0_expected_random", "r3_expected_random"]
        estimates, draws = cluster_bootstrap(tie_free, columns, boot, seed + 40 + i)
        delta = draws[:, 1] - draws[:, 0]
        rows.append({
            "benchmark": bench,
            "rule": "tie_free",
            "acc_r0": estimates[columns[0]][0],
            "acc_r3": estimates[columns[1]][0],
            "delta": estimates[columns[1]][0] - estimates[columns[0]][0],
            "lo": float(np.percentile(delta, 2.5)),
            "hi": float(np.percentile(delta, 97.5)),
            "tie_rate_r0": 0.0,
            "tie_rate_r3": 0.0,
            "n_team_question": len(tie_free),
        })
    return pd.DataFrame(rows)

def pvalue(value: float) -> str:
    return f"{value:.1e}" if value < 0.001 else f"{value:.3f}"

def write_visual_tex(summary: pd.DataFrame, path: Path) -> None:
    labels = {"damage": "Damage", "recovery_failure": "Recovery failure"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\caption{HallusionBench visual-dependence audit using existing trajectories only. Damage denotes the loss of an initially correct team decision; recovery failure denotes persistence of an initially incorrect decision. MH odds ratios control for item difficulty using strata of with-image R0 accuracy.}",
        r"\label{tab:visual-language-existing}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Outcome & VD & VS & Crude OR & MH OR [95\\% CI] & CMH $p$ \\\\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{labels[row.outcome]} & {row.vd_rate:.3f} ($n={row.vd_n:,}$) "
            f"& {row.vs_rate:.3f} ($n={row.vs_n:,}$) & {row.crude_or:.2f} "
            f"& \\textbf{{{row.mh_or:.2f} [{row.lo:.2f}, {row.hi:.2f}]}} "
            f"& ${pvalue(row.cmh_p)}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def write_baseline_tex(summary: pd.DataFrame, path: Path) -> None:
    methods = [
        ("Best single (one call)", "best_single", "1"),
        ("R0 majority", "r0_majority", "3"),
        ("Rank-weighted R0", "rank_weighted", "3"),
        ("SC@6 (best single)", "sc6_global_best", "6"),
        ("One-round debate", "r1_majority", "6"),
        ("R3 debate", "r3_majority", "12"),
        ("R0 oracle", "r0_oracle", "3 candidates"),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Accuracy (\%) of debate and cheaper selection baselines on the same R3/C-02 analysis universe. Confidence intervals and paired differences are provided in the accompanying CSV. The R0 oracle is an unattainable upper bound and is not included in cost comparisons.}",
        r"\label{tab:cheap-selection-r3}", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        "Method & Gen. & HallusionBench & MMStar & MMMU & MME-CoT \\\\", r"\midrule",
    ]
    for label, method, generations in methods:
        values = [f"{100 * summary[(summary.benchmark == bench) & (summary.method == method)].iloc[0].accuracy:.2f}"
                  for bench in BENCHES]
        lines.append(f"{label} & {generations} & " + " & ".join(values) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def write_tie_tex(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "expected_random": "Expected random",
        "tie_as_wrong": "Tie as wrong",
        "tie_as_correct": "Tie as correct",
        "rank_based": "Rank-based",
        "tie_free": "Tie-free subset",
    }
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Sensitivity of the R0-to-R3 accuracy change to the tie-handling rule. Cells report percentage-point changes with question-clustered 95\% confidence intervals. The tie-free analysis uses a fixed subset with no tie at either endpoint.}",
        r"\label{tab:tie-sensitivity-r3}", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccc}", r"\toprule",
        "Rule & HallusionBench & MMStar & MMMU & MME-CoT \\\\", r"\midrule",
    ]
    for rule, label in labels.items():
        values = []
        for bench in BENCHES:
            row = summary[(summary.benchmark == bench) & (summary.rule == rule)].iloc[0]
            values.append(f"{100 * row.delta:.2f} [{100 * row.lo:.2f}, {100 * row.hi:.2f}]")
        lines.append(label + " & " + " & ".join(values) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=RAW)
    parser.add_argument("--out", default=DATA)
    parser.add_argument("--tex-out", default=None)
    parser.add_argument("--visual-round", type=int, default=4)
    parser.add_argument("--endpoint", type=int, default=3)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    tex_out = Path(args.tex_out) if args.tex_out else out
    out.mkdir(parents=True, exist_ok=True)
    tex_out.mkdir(parents=True, exist_ok=True)

    columns = ["benchmark", "question_id", "mode", "team_id", "model", "round",
               "answer", "is_correct", "gold", "category"]
    raw = pd.read_csv(args.csv, dtype=str, keep_default_na=False,
                      encoding="utf-8-sig", usecols=columns)
    raw["round"] = raw["round"].astype(int)
    raw["is_correct"] = raw["is_correct"].astype(float)

    visual = visual_summary(visual_panel(raw, args.visual_round))
    panel = audit_panel(raw, args.endpoint)
    baselines = baseline_summary(panel, args.boot, args.seed)
    ties = tie_summary(panel, args.boot, args.seed)

    visual.to_csv(out / "supE_visual_language.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(out / "supF_audit_panel_r3_c02.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(out / "supF_baseline_selection_r3.csv", index=False, encoding="utf-8-sig")
    ties.to_csv(out / "supG_tie_sensitivity_r3.csv", index=False, encoding="utf-8-sig")
    write_visual_tex(visual, tex_out / "table_visual_language_existing.tex")
    write_baseline_tex(baselines, tex_out / "table_baseline_selection_r3.tex")
    write_tie_tex(ties, tex_out / "table_tie_sensitivity_r3.tex")
    print(f"Wrote Section 1.2, 1.6, and 1.7 results to {out}")

if __name__ == "__main__":
    main()
