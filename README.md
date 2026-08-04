# Supplementary Material

Code and data for Madness Is Already in the Room: A Black-Box Analysis of Answer Transitions.

Everything needed to reproduce the tables and figures of the main paper is included.
No component is withheld pending acceptance.

```
supplementary_material/
├── code_debate/               Experiment: runs the 3-agent debates, emits raw responses
├── code_judge/                Scoring: equivalence judge for open-ended answers
├── code_results_analysis/     Analysis: raw responses -> tables and figures
└── data_results_analysis/     One source table + the derived file behind each result
```

File names carry their destination in the paper: `t3_` = Table 3, `f2_` = Figure 2,
`s45_` = Section 4.5, `sup` = an analysis reported only in this supplement.
Scripts are numbered in execution order.

---

## 1. Quick start

The analysis reproduces from the shipped source table; no GPU is required.

```bash
cd code_results_analysis
pip install pandas numpy scipy matplotlib

python3 step1_outcomes.py              # Table 3, Figure 1, Figure 2   (Sec 4.1, 4.2)
python3 step2_situation_survival.py    # Table 4, Table 5              (Sec 4.3, 4.4)
python3 step3_agent_events.py          # intermediate -> step 4        (Sec 4.4)
python3 step4_team_composition.py      # Table 6                       (Sec 4.4)
python3 step5_round_dynamics.py        # Sec 4.6 (no float)
python3 step6_visual_panel_hb.py       # intermediate -> step 8        (Sec 4.5)
python3 step7_visual_panel_mmstar.py   # intermediate -> step 8        (Sec 4.5)
python3 step8_visual_axis.py           # Sec 4.5 figures, input to Figure 3
python3 step9_make_fig1_fig2.py        # Figure 1, Figure 2
python3 step10_make_fig3.py            # Figure 3

python3 step11_sup_opportunity.py      # Supplement: opportunity confounding
python3 step12_sup_rank_regression.py  # Supplement: rank regression + cross-fitting
python3 step13_sup_visual_robustness.py # Supplement: Breslow-Day + standardization
python3 step14_sup_freeform.py         # Supplement: open-ended scoring
python3 step15_sup_critique_robustness.py # Supplement 1.2, 1.6, 1.7 tables
python3 step16_full_round_tables.py    # Full round and HallusionBench tables
python3 make_supp_tex.py               # Emits the supplement's LaTeX sections
```

Steps 1--10 reproduce the main paper. Steps 11--16 produce the additional
analyses and full-results tables reported in the supplement. Step 13 requires
steps 6 and 7 first, and step 14 reads `../code_judge/equiv_cache.json`.
Step 15 reproduces the original R0--R4 HallusionBench audit in Section 1.2 and
the R3/C-02 analyses in Sections 1.6--1.7. Step 16 uses expected uniform
tie-breaking for the cross-benchmark table and the experiment's seeded uniform
tie choice for the nonlinear grouped HallusionBench metrics.

No arguments are needed. Paths resolve from the script location via `_paths.py`:

| Variable | Default | Override |
|---|---|---|
| `DATA` | `../data_results_analysis` | `DATA_DIR` |
| `RAW`  | `DATA/source_responses.csv.gz` | `SRC` |
| `FIGS` | `../figures` | `FIG_DIR` |
| `R_last` | round 3 | `RLAST` |

Every script also accepts `--csv`, `--data`, and `--out` explicitly. Total runtime is
roughly 25 minutes on a laptop; steps 1 and 2 dominate. Figures land in `../figures/`.

`_paths.py` (path resolution), `_figstyle.py` (figure style) and `_stats.py`
(Mantel--Haenszel, Breslow--Day, direct standardization) are modules, not steps;
they are imported, not run. `statsmodels` is additionally required by step 12.

---

## 2. Data

### 2.1 Source table

`source_responses.csv.gz` is the single source for every number in the paper.
One row is one agent response.

| Column | Meaning |
|---|---|
| `benchmark` | hallusionbench / mmstar / mmmu / mme-cot |
| `question_id` | question identifier |
| `mode` | `debate`, `self_reflect`, or `blind` |
| `team_id` | one of ten 3-agent teams; `single` for `self_reflect` |
| `model` | Gemma-4 / InternVL / Llama-3.2 / Pixtral / Qwen3.5 |
| `round` | 0–4 |
| `answer` | extracted answer string (may be empty) |
| `answer_type` | mcq / binary / freeform |
| `is_correct` | 0/1 |
| `gold` | reference answer |
| `parse_ok`, `flag` | extraction metadata |
| `category`, `category_detail` | benchmark-provided category labels |
| `split` | MMMU only |

The table also contains a `blind` condition — a grey-image Round 0 control — that
is not analyzed in the paper. It is retained because the empirical answer space
used for the chance-adoption baseline in Figure 2 is estimated over all observed
responses, so removing it would change that baseline.

`panel_team_question.csv.gz` is the team–question panel derived from it: one row per
(team, question), carrying the round-0 state, `b`, `f`, and `delta`. Most downstream
steps read this rather than the source table.

### 2.2 Which file backs which result

| File | Paper | Produced by |
|---|---|---|
| `t3_gain_loss.csv` | Table 3 | step 1 |
| `t3f1_by_state.csv` | Table 3, Figure 1 | step 1 |
| `f2_adoption.csv` | Figure 2, Sec 4.2 | step 1 |
| `f2_adoption_baseline.csv` | Figure 2 | step 1 |
| `s41_selfreflect_control.csv` | Sec 4.1 control | step 1 |
| `panel_team_question.csv.gz` | (panel used by steps 4, 8) | step 1 |
| `t4_situation_response.csv` | Table 4 | step 2 |
| `t5_survival.csv` | Table 5 | step 2 |
| `t5_survival_sensitivity.csv` | Table 5 note | step 2 |
| `s46_round_trend.csv` | Sec 4.6 | step 2 |
| `agent_events.csv` | (intermediate) | step 3 |
| `t6_team_composition.csv` | Table 6 | step 4 |
| `s46_error_correlation.csv` | Sec 4.6 | step 5 |
| `visual_panel_hb.csv` | (intermediate) | step 6 |
| `visual_panel_mmstar.csv` | (intermediate) | step 7 |
| `s45_mh_or.csv` | Sec 4.5 | step 8 |
| `f3_or_by_stratum.csv` | Figure 3 | step 8 |
| `clean_question_ids_*.json` | sampled question identifiers | — |
| `supA_opportunity.csv` | Supplement, opportunity | step 11 |
| `supB_rank_regression.csv` | Supplement, rank regression | step 12 |
| `supB_rank_crossfit.csv` | Supplement, cross-fitted rank | step 12 |
| `supB_holder_events.csv` | (holder-level unit table) | step 12 |
| `supC_visual_robustness.csv` | Supplement, visual axis | step 13 |
| `supD_freeform.csv` | Supplement, open-ended scoring | step 14 |
| `supE_visual_language.csv` | Supplement 1.2, visual-language audit | step 15 |
| `supF_audit_panel_r3_c02.csv` | Supplement 1.6--1.7 analysis panel | step 15 |
| `supF_baseline_selection_r3.csv` | Supplement 1.6, cheap baselines | step 15 |
| `supG_tie_sensitivity_r3.csv` | Supplement 1.7, tie handling | step 15 |
| `full_round_accuracy.csv` | Full round-wise results | step 16 |
| `hallusionbench_round_metrics.csv` | Full HallusionBench metrics | step 16 |
| `table_visual_language_existing.tex` | Supplement 1.2 table | step 15 |
| `table_baseline_selection_r3.tex` | Supplement 1.6 table | step 15 |
| `table_tie_sensitivity_r3.tex` | Supplement 1.7 table | step 15 |
| `table_full_round_accuracy.tex` | Full cross-benchmark table | step 16 |
| `table_hb_round_metrics.tex` | Full HallusionBench table | step 16 |

Files prefixed `extra_` are additional breakdowns the scripts emit but the paper
does not use (per-team and per-model splits, weighted-voting variants).

### 2.3 Column names

Some derived CSVs retain non-English column labels from the original analysis.
The scripts consume these labels directly, so reproduction requires no renaming;
the table below is for reading the files by hand.

| Label | Meaning |
|---|---|
| `지표` / `값` | metric / value |
| `구분` / `단위` / `계열` / `판본` / `비고` | group / unit / series / variant / note |
| `상황` | Round-0 state (MaC, MiC, CC, MaW, CW) |
| `비율`, `비율_lo`, `비율_hi` | share of team--questions, with interval |
| `b평균` / `f평균` / `delta평균` | mean $b$ / mean $f$ / mean $\Delta$ |
| `이득` / `손실` | gain ($\Delta>0$) / loss ($\Delta<0$) |
| `이득률` / `손실률` | gain rate / loss rate |
| `채택` / `창발` | adoption / generation |
| `채택_정답`, `채택_오답`, `창발_정답`, `창발_오답` | adoption or generation, crossed with correct or wrong |
| `채택률` | adoption rate |
| `답공간K_중앙값` | median size of the empirical answer space $K$ |
| `R0풀크기_평균` | mean size of the Round-0 answer pool |
| `우연채택기준선`, `기준선_lo`, `기준선_hi` | chance-adoption baseline, with interval |
| `실제채택률`, `채택_lo`, `채택_hi` | observed adoption rate, with interval |
| `초과` | observed adoption minus the chance baseline |
| `노출` / `수용` / `유지` / `신규` | exposure / acceptance / retention / novel response |
| `유지율` | retention rate |
| `기준선` | no-debate baseline for the state (0 or 1/3) |
| `생존율_기대` / `생존율_이벤트` | survival, expected score / binarized event |
| `MaC_손실기대`, `MaC_손실발생률` | MaC loss, expected score / event rate |
| `MiC_구제기대`, `MiC_구제발생률` | MiC recovery, expected score / event rate |
| `CC_손실발생률`, `CC_구제발생률`, `CC_순변화` | CC loss rate / recovery rate / net change |
| `MaW_구제기대`, `MaW_구제발생률`, `CW_구제기대`, `CW_구제발생률` | same quantities for MaW and CW |
| `MaC손실` / `MiC구제` | per-team MaC loss rate / MiC recovery rate |
| `팀정확도` / `팀Δ` | team accuracy / team-level change |
| `강도격차` | ability gap |
| `위험차` / `기저이동` | risk difference / baseline shift |
| `logOR_대조1` / `logOR_방향성` | log odds ratio, contrast 1 / directionality |
| `s_mean` | mean member ability |
| `에이전트` / `팀` | agent / team |
| `debate_팀다수결` / `debate_에이전트평균` | debate team majority / debate agent mean |
| `..._mcq만` | same series restricted to multiple-choice questions |
| `R1만` | Round-1-only variant |
| `전체` | all (no restriction) |
| `정답\|채택`, `정답\|창발` | correct given adoption / correct given generation |

A trailing `_lo` or `_hi` marks a 95% interval bound, and a trailing `_r` marks a
within-benchmark rank.

Files are UTF-8 with BOM (`encoding="utf-8-sig"`).

---

## 3. Conventions used throughout

- Endpoint: `R_last = R3`, the first round by which at least 90% of cumulative
  answer changes have occurred. Set `RLAST` to reproduce other endpoints.
- Scoring: `b` and `f` are the expected correctness of the majority vote under
  uniform tie-breaking, `1[gold in T] / |T|`, where `T` is the set of most-voted
  answers. `delta = f - b`.
- Exclusions: 616 team–questions lacking a complete set of three valid answers
  at round 0 or round 3 are dropped, leaving 41,494 of 42,110.
- Confidence intervals: Question-cluster bootstrap, B = 2000, seed 42.
  Comparisons between two quantities use identical resample indices.
- Difficulty stratification: Leave-team-out difficulty is the accuracy over the 12
  round-0 responses produced for that question by the two models not on the team.

---

## 4. Experiment code

`code_debate/` runs the debates and writes raw model responses. It requires five vLLM
servers (one per model family); see `code_debate/README.md` and `configs/debate.yaml`,
which records the exact serving and sampling settings used (temperature 0.7, top-p 0.9,
seed 42, and the ten team compositions).

`prompts/role_prompts.py` and `src/prompt_builder.py` together define the prompts.
The debate and self-reflection conditions share this code and diverge at a single line
in `main.py`: self-reflection omits peer responses from the previous-round block. The
image and question are re-supplied every round; only the immediately preceding round's
responses are passed, labelled self versus other.

Running the debates end to end takes substantial GPU time. The analysis in Section 1
does not require it — `source_responses.csv.gz` is the output of that stage.

---

## 5. Open-ended scoring

`code_judge/` contains the equivalence judge used for MME-CoT's open-ended
questions; multiple-choice answers are scored by option-letter matching and never
reach it. The final answer is extracted by the debate runner's rule-based
`parse_final`, normalized, and compared with the reference; only pairs that fail
that comparison are sent to the judge, which answers yes or no at temperature 0.

Verdicts are cached in `code_judge/equiv_cache.json`, keyed on the raw
(reference, prediction) pair. The cache produced by this experiment is included
(8,776 pairs), so open-ended scoring reproduces without running the judge model.
Running the judge is only necessary for new responses and requires a served
Qwen3-32B; see `code_judge/README.md`.

## 6. Benchmarks

All four are public. We sample single-image instances while preserving each benchmark's
original category distribution; the retained identifiers are listed in
`clean_question_ids_*.json`.

| Benchmark | Items used |
|---|---|
| HallusionBench | 781 (binary) |
| MMStar | 1,355 (multiple choice) |
| MMMU | 1,500 (multiple choice) |
| MME-CoT | 575 (multiple choice + open-ended) |

Benchmark images and questions are distributed by their original authors under their
own licenses and are not redistributed here. `code_debate/data_prep/` contains the
conversion and validation code for all four benchmarks.

---

## 7. Known limitations

- A single decoding seed (42) was used. Reported intervals reflect question-level
  sampling variability, not seed variability.
- Model sizes are restricted to the 9B–14B range.
- Only answers are analyzed; the reasoning text exchanged during debate is not.
