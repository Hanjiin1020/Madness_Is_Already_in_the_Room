#!/usr/bin/env python3
"""Build the two round-wise full-results tables."""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, RAW  # noqa: E402

BENCHES = ["hallusionbench", "mmstar", "mmmu", "mme-cot"]
BENCH_LABELS = {
    "hallusionbench": "HallusionBench",
    "mmstar": "MMStar",
    "mmmu": "MMMU",
    "mme-cot": "MME-CoT",
}
ROW_END = r" \\"

def expected_correct(answers: list[str], gold: str) -> float:
    valid = [answer for answer in answers if answer]
    if not valid:
        return 0.0
    counts = Counter(valid)
    peak = max(counts.values())
    top = [answer for answer, count in counts.items() if count == peak]
    return float(gold in top) / len(top)

def figure_key(question_id: str) -> str:
    return question_id.rsplit("_", 1)[0]

def question_pair_key(question_id: str) -> str:
    parts = question_id.split("_")
    return "_".join(parts[:-2] + [parts[-1]]) if len(parts) >= 3 else question_id

def grouped_accuracy(correct: dict[str, bool],
                     groups: dict[str, list[str]]) -> float:
    if not groups:
        return np.nan
    return float(np.mean([
        all(correct.get(question, False) for question in questions)
        for questions in groups.values()
    ]))

def random_tie_correct(answers: list[str], gold: str,
                       rng: random.Random) -> bool:
    valid = [answer for answer in answers if answer]
    if not valid:
        return False
    counts = Counter(valid)
    peak = max(counts.values())
    top = sorted(answer for answer, count in counts.items() if count == peak)
    selected = top[0] if len(top) == 1 else rng.choice(top)
    return selected == gold

def infer_model_key(raw: pd.DataFrame) -> dict[str, str]:
    debate = raw[raw["mode"] == "debate"]
    team_models = {
        team: set(group.model) for team, group in debate.groupby("team_id")
    }
    letters = sorted(set("".join(team_models)))
    mapping = {}
    for letter in letters:
        candidates = [models for team, models in team_models.items() if letter in team]
        common = set.intersection(*candidates) if candidates else set()
        if len(common) == 1:
            mapping[letter] = common.pop()
    return mapping

def build_scores(raw: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    debate = raw[raw["mode"] == "debate"].copy()
    records = []
    for (bench, team, question, round_), group in debate.groupby(
            ["benchmark", "team_id", "question_id", "round"], sort=False):
        records.append({
            "benchmark": bench,
            "team_id": team,
            "question_id": question,
            "round": int(round_),
            "score": expected_correct(group.answer.tolist(), group.gold.iloc[0]),
        })
    question_scores = pd.DataFrame(records)
    full = (question_scores.groupby(["benchmark", "team_id", "round"], as_index=False)
            .score.mean().rename(columns={"score": "accuracy"}))

    hb = debate[debate.benchmark == "hallusionbench"].copy()
    figures: dict[str, list[str]] = defaultdict(list)
    pairs: dict[str, list[str]] = defaultdict(list)
    questions = list(pd.unique(hb.question_id))
    for question in questions:
        figures[figure_key(question)].append(question)
        pairs[question_pair_key(question)].append(question)

    grouped = {
        key: group for key, group in hb.groupby(
            ["team_id", "round", "question_id"], sort=False)
    }
    rng = random.Random(seed)
    hb_rows = []
    for team in sorted(hb.team_id.unique()):
        for round_ in sorted(hb[hb.team_id == team]["round"].unique()):
            correct = {}
            for question in questions:
                group = grouped.get((team, round_, question))
                if group is None:
                    continue
                correct[question] = random_tie_correct(
                    group.answer.tolist(), group.gold.iloc[0], rng)
            aacc = float(np.mean(list(correct.values()))) if correct else np.nan
            facc = grouped_accuracy(correct, figures)
            qacc = grouped_accuracy(correct, pairs)
            hb_rows.append({
                "team_id": team,
                "round": int(round_),
                "aAcc": aacc,
                "fAcc": facc,
                "qAcc": qacc,
                "Avg": float(np.mean([aacc, facc, qacc])),
            })
    return full, pd.DataFrame(hb_rows)

def format_value(value: float, best: float) -> str:
    rendered = f"{100 * value:.2f}"
    return rf"\textbf{{{rendered}}}" if np.isclose(value, best) else rendered

def write_full_tex(summary: pd.DataFrame, path: Path,
                   model_key: dict[str, str] | None = None) -> None:
    teams = sorted(summary.team_id.unique())
    rounds = sorted(summary["round"].unique())
    maxima = summary.groupby(["benchmark", "round"]).accuracy.max()
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{Round-wise majority-vote accuracy (\%) for {len(teams)} heterogeneous compositions across the evaluated benchmarks. Ties are handled by the expected accuracy under uniform random tie-breaking. The best result in each benchmark--round column is shown in bold.}}",
        r"\label{tab:full-round-accuracy}", r"\setlength{\tabcolsep}{2.4pt}",
        r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{l*{{{len(BENCHES) * len(rounds)}}}{{r}}}}",
        r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{{len(rounds)}}}{{c}}{{{BENCH_LABELS[bench]}}}"
                             for bench in BENCHES) + ROW_END,
    ]
    start = 2
    for _ in BENCHES:
        stop = start + len(rounds) - 1
        lines.append(rf"\cmidrule(lr){{{start}-{stop}}}")
        start = stop + 1
    lines.append("Composition & " + " & ".join(
        f"R{round_}" for _ in BENCHES for round_ in rounds) + ROW_END)
    lines.append(r"\midrule")
    indexed = summary.set_index(["benchmark", "team_id", "round"])
    for team in teams:
        values = []
        for bench in BENCHES:
            for round_ in rounds:
                value = indexed.loc[(bench, team, round_), "accuracy"]
                values.append(format_value(value, maxima.loc[(bench, round_)]))
        lines.append(team + " & " + " & ".join(values) + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}%", r"}"]
    if model_key:
        key = ", ".join(f"{letter} = {model}" for letter, model in model_key.items())
        lines += [r"\vspace{2pt}", "",
                  rf"\parbox{{\textwidth}}{{\footnotesize \textit{{Composition key:}} {key}.}}"]
    lines += [r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def write_hb_tex(summary: pd.DataFrame, path: Path) -> None:
    teams = sorted(summary.team_id.unique())
    rounds = sorted(summary["round"].unique())
    metrics = ["aAcc", "fAcc", "qAcc", "Avg"]
    maxima = {metric: summary.groupby("round")[metric].max() for metric in metrics}
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{Round-wise HallusionBench performance (\%) for {len(teams)} heterogeneous compositions. \textit{{aAcc}} is per-question accuracy; \textit{{fAcc}} and \textit{{qAcc}} require all questions in a figure or original/edited question group to be correct; \textit{{Avg}} is their unweighted mean. Ties use a seeded uniform random choice. The best result in each metric--round column is shown in bold.}}",
        r"\label{tab:hb-full-round-metrics}", r"\setlength{\tabcolsep}{2.4pt}",
        r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{l*{{{len(metrics) * len(rounds)}}}{{r}}}}",
        r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{{len(rounds)}}}{{c}}{{{metric}}}"
                             for metric in metrics) + ROW_END,
    ]
    start = 2
    for _ in metrics:
        stop = start + len(rounds) - 1
        lines.append(rf"\cmidrule(lr){{{start}-{stop}}}")
        start = stop + 1
    lines.append("Composition & " + " & ".join(
        f"R{round_}" for _ in metrics for round_ in rounds) + ROW_END)
    lines.append(r"\midrule")
    indexed = summary.set_index(["team_id", "round"])
    for team in teams:
        values = []
        for metric in metrics:
            for round_ in rounds:
                value = indexed.loc[(team, round_), metric]
                values.append(format_value(value, maxima[metric].loc[round_]))
        lines.append(team + " & " + " & ".join(values) + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=RAW)
    parser.add_argument("--out", default=DATA)
    parser.add_argument("--tex-out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    tex_out = Path(args.tex_out) if args.tex_out else out
    out.mkdir(parents=True, exist_ok=True)
    tex_out.mkdir(parents=True, exist_ok=True)
    columns = ["benchmark", "question_id", "mode", "team_id", "model", "round",
               "answer", "gold"]
    raw = pd.read_csv(args.csv, dtype=str, keep_default_na=False,
                      encoding="utf-8-sig", usecols=columns)
    raw["round"] = raw["round"].astype(int)
    full, hallusion = build_scores(raw, args.seed)
    full.to_csv(out / "full_round_accuracy.csv", index=False, encoding="utf-8-sig")
    hallusion.to_csv(out / "hallusionbench_round_metrics.csv", index=False,
                     encoding="utf-8-sig")
    write_full_tex(full, tex_out / "table_full_round_accuracy.tex", infer_model_key(raw))
    write_hb_tex(hallusion, tex_out / "table_hb_round_metrics.tex")
    print(f"Wrote full round results to {out}")

if __name__ == "__main__":
    main()
