# Open-Ended Answer Equivalence Judge

This directory contains the equivalence judge used to score open-ended debate
answers. Multiple-choice responses are scored by option-letter matching and do
not enter this pipeline.

## Motivation

Exact string matching can reject semantically equivalent open-ended answers.

| Gold | Prediction | Exact match | Semantic result |
|---|---|---|---|
| `14/3` | `\boxed{\dfrac{14}{3}}` | No | Equivalent |
| `2 cm` | `2` | No | Equivalent under the judge policy |

The Qwen3-32B judge determines whether the reference and prediction are
equivalent, and each verdict is stored in a cache.

## Structure

```text
.
├── judge_freeform.py       # Single-process judging
├── judge_shard.py          # Sharded judging
├── merge_cache.py          # Cache merge
├── serve_judge.sh          # Qwen3-32B server
├── equiv_cache.json        # Included verdict cache
└── src/
    ├── score_equiv_judge.py   # Equivalence scorer
    └── score_llm_judge.py     # Answer normalization
```

## Processing flow

```text
debate.jsonl containing open-ended questions
  └─ parse_final() extracts the answer with deterministic rules
       └─ unique reference and prediction pairs
            ├─ normalized exact matches are resolved locally
            └─ unresolved pairs are sent to Qwen3-32B
                 └─ equiv_cache.json stores Boolean verdicts
```

The judge does not extract answers. The pipeline reuses `parse_final()` from the
debate code so that answer extraction has a single implementation. The cache is
incremental, and reruns skip previously judged pairs. Failed judge calls are
conservatively recorded as incorrect.

## Installation

The judge client reuses the debate client dependencies. The judge server uses
the main serving environment because its vLLM, PyTorch, and Transformers stack
matches the debate serving stack.

```bash
# Judge client
python -m venv .venv && source .venv/bin/activate
pip install -r ../code_debate/requirements.txt

# Judge server
python -m venv .venv-serving && source .venv-serving/bin/activate
pip install -r ../code_debate/requirements-serving.txt
```

## Single-server execution

```bash
# Start the server
GPUS=0,1 PORT=8006 bash serve_judge.sh

# Run judging
python judge_freeform.py \
  --runs-dir ../code_debate/runs \
  --judge-base http://localhost:8006/v1 \
  --cache equiv_cache.json
```

## Two-server sharded execution

```bash
GPUS=0,1 PORT=8006 bash serve_judge.sh &
GPUS=2,3 PORT=8007 bash serve_judge.sh &

python judge_shard.py --shard 0 --nshards 2 --runs-dir ../code_debate/runs \
  --judge-base http://localhost:8006/v1 --out-cache shard0.json &
python judge_shard.py --shard 1 --nshards 2 --runs-dir ../code_debate/runs \
  --judge-base http://localhost:8007/v1 --out-cache shard1.json &
wait

python merge_cache.py --cache equiv_cache.json shard0.json shard1.json
```

Pairs are assigned by `md5(key) % nshards`, so shards do not overlap. Each
worker writes only its own output file and reads the shared cache, avoiding a
write race.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--runs-dir` | Required | Root containing debate result files |
| `--cache` | `equiv_cache.json` | Incremental verdict cache |
| `--debate-code` | `../code_debate` | Location providing `parse_final()` |
| `--judge-base` | `http://localhost:8006/v1` | Judge endpoint |
| `--judge-model` | `Qwen/Qwen3-32B` | Judge model identifier |
| `--max-conc` | 64 | Maximum concurrent requests |

The serving script accepts `GPUS`, `PORT`, `MAXLEN`, `GPU_UTIL`, `VLLM_BIN`, and
`HF_HUB_OFFLINE`.

## Included cache

`equiv_cache.json` maps a tab-separated reference and prediction pair to a
Boolean verdict. Downstream scoring consults the cache before normalized string
matching.

The included experiment cache contains 8,776 pairs: 862 equivalent pairs and
7,914 non-equivalent pairs. It reproduces the existing open-ended scores without
starting the judge server. New responses require judging only when their pairs
are absent from the cache.

## Operational notes

- `src/score_llm_judge.py` also contains an LLM-based answer extractor, but this
  pipeline uses the deterministic `parse_final()` output.
- Tensor-parallel serving with two GPUs uses `NCCL_P2P_DISABLE=1` and
  `--disable-custom-all-reduce`, as configured in `serve_judge.sh`.
