# MLLM Multi-Agent Debate Code

This directory contains the experiment code for heterogeneous three-agent MLLM
debates. It generates model responses in JSONL format. Scoring and analysis are
handled by the adjacent submission directories.

## Structure

```text
.
├── main.py                 # Debate entry point
├── requirements.txt        # Client dependencies
├── requirements-serving.txt         # Main serving environment
├── requirements-serving-mllama.txt  # Llama serving environment
├── prompts/
│   └── role_prompts.py     # Prompt content
├── src/
│   ├── prompt_builder.py   # Prompt assembly
│   └── mllm_backend.py     # Model client and answer parsing
├── configs/
│   └── debate.yaml         # Experiment configuration
├── data_prep/
│   ├── _convert_common.py  # Shared validation and image normalization
│   ├── convert_hallusionbench.py
│   ├── convert_mmstar.py
│   ├── convert_mmmu.py
│   ├── convert_mme-cot.py
│   └── CONVERSION_NOTE.md
├── serve_*.sh, _common.sh  # Model servers
└── runs/                   # Outputs
```

## Execution modes

| Mode | Previous-round input | Rounds | Agents |
|---|---|---:|---:|
| `debate` | Own response and peer responses | 5 (R0--R4) | 3 |
| `self_reflect` | Own response only | 5 (R0--R4) | 1 |

Both modes use the same code and prompt templates. They diverge only when the
previous-round response block is selected in `main.py`.

```python
# Self-reflection receives only its own previous response.
prev_fn = prev_block_self if mode == "self_reflect" else prev_block_selfother
```

The self-reflection control removes peer responses while retaining the prompt
structure, number of rounds, and decoding configuration.

## Models

| `model_key` | Model | Default port |
|---|---|---:|
| `gemma4` | Gemma-4-12B-it | 8003 |
| `qwen35` | Qwen3.5-9B | 8004 |
| `internvl3_5` | InternVL3.5-14B | 8001 |
| `pixtral` | Pixtral-12B-2409 | 8011 |
| `llama32vision` | Llama-3.2-11B-Vision | 8002 |

The experiment evaluates the ten three-model combinations defined in
`configs/debate.yaml`.

## Prompts

Round 0 uses the following structure.

```text
[SYSTEM] You are a vision-language agent answering a question about the given image.
         Look at the image carefully and reason step by step before answering.
[USER]   Question: <QUESTION>
         Options:
         A. ... / B. ...

         <ANSWER_FORMAT>
         First, think step by step ... write exactly 'Answer: ' followed by your final answer.
         <IMAGE>
```

The options block is included only for multiple-choice questions. From Round 1
onward, the prompt also includes the previous-round block and a reconsideration
instruction. `src/prompt_builder.py` selects the instruction from the presence
of the `[OTHER AGENTS' PREVIOUS RESPONSES]` marker.

The converted `answer_type` field selects the answer-format instruction.

| Benchmark or subset | `answer_type` | Required final answer |
|---|---|---|
| HallusionBench | `binary` | Exactly `Yes` or `No` |
| MMStar | `mcq` | One option letter |
| MMMU | `mcq` | One option letter |
| MME-CoT multiple choice | `mcq` | One option letter |
| MME-CoT open ended | `open` | A short phrase, number, or expression |

`main.py` reads this field automatically. For older converted files without the
field, option columns identify MCQ rows, and the HallusionBench VD/VS label with
a yes/no reference identifies binary rows. `--answer-mode` remains available as
an explicit override.

- A block containing the marker uses `DEBATE_INSTRUCTION_SELFOTHER`.
- A block without the marker uses `SELF_REFLECT_INSTRUCTION`.

The prompt variants can be inspected with this command.

```bash
python src/prompt_builder.py
```

## Installation

Three environments are used because Llama-3.2-Vision requires a different vLLM
version from the other models.

```bash
# Client environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Main serving environment
python -m venv .venv-serving && source .venv-serving/bin/activate
pip install -r requirements-serving.txt

# Llama serving environment
python -m venv .venv-mllama && source .venv-mllama/bin/activate
pip install -r requirements-serving-mllama.txt
```

## Data preparation

```bash
python data_prep/convert_hallusionbench.py \
  --src /path/to/HallusionBench.parquet

python data_prep/convert_mmstar.py \
  --src /path/to/mmstar.parquet

python data_prep/convert_mmmu.py \
  --src /path/to/MMMU/*/*.parquet --per-subject 50 --seed 42

python data_prep/convert_mme-cot.py \
  --src /path/to/MME-CoT_single_opt.parquet \
        /path/to/MME-CoT_single_test.parquet
```

All converters emit the common `index`, `question`, `image`, `answer`, and
optional `A`--`L` interface. Each converter applies its shipped clean-ID list by
default. A caller can override the list with `--ids`. MMMU uses balanced
per-subject sampling only when no ID list is supplied. Source schemas and
validation rules are documented in `data_prep/CONVERSION_NOTE.md`.

## Model serving

Activate the main serving environment before starting Gemma, Qwen, InternVL,
and Pixtral.

```bash
MAXLEN=32768 GPU=0 PORT=8004 bash serve_qwen.sh
MAXLEN=32768 GPU=1 PORT=8003 bash serve_gemma.sh
MAXLEN=32768 GPU=2 PORT=8001 bash serve_internvl.sh
MAXLEN=32768 GPU=3 PORT=8011 bash serve_pixtral.sh
```

Start Llama from its separate environment.

```bash
VLLM_BIN=$PWD/.venv-mllama/bin/vllm MAXLEN=32768 GPU=0 PORT=8002 bash serve_llama.sh
```

The serving scripts accept `GPU`, `PORT`, `MAXLEN`, `GPU_UTIL`, `VLLM_BIN`, and
`LOGDIR`. Their default values are recorded in the scripts. For cross-node
execution, set `VLLM_URL_<model_key>` to the externally reachable vLLM endpoint.

## Debate execution

The following command runs a three-agent, five-round debate.

```bash
python main.py --parquet data/MMStar_converted.parquet \
  --agents gemma4 internvl3_5 qwen35 --mode debate --rounds 5 \
  --temperature 0.7 --top-p 0.9 --seed 42 --max-tokens 16384 \
  --max-records-in-flight 20 --out-dir runs --run-id mmstar_GIQ
```

The following command runs the single-model self-reflection control.

```bash
python main.py --parquet data/MMStar_converted.parquet \
  --agents qwen35 --mode self_reflect --rounds 5 \
  --temperature 0.7 --top-p 0.9 --seed 42 --max-tokens 16384 \
  --max-records-in-flight 20 --out-dir runs --run-id selfref_qwen35_mmstar
```

Llama uses a 4,096-token output cap and a minimum-token setting to limit
repetitive divergence.

```bash
--min-tokens 128 --min-tokens-models llama32vision
```

Per-model output limits can be set with `MAXTOK_<model_key>=N`. The complete
configuration is stored in `configs/debate.yaml`.

## Outputs

`runs/<run-id>/debate.jsonl` contains one sample per line.

```json
{
  "sample_id": "...", "qtype": "mcq|open|binary", "gt": "...", "valid": ["A","B","C","D"],
  "final_answers": {"<agent>": "..."},
  "transcript": [
    {"round": 0, "outputs": {"<agent>": {"response": "...", "finish": "stop", "tokens": 123}}}
  ]
}
```

`runs/<run-id>/run_meta.json` records the execution configuration.

`parse_final()` in `src/mllm_backend.py` extracts the final answer through an
explicit `Answer:` line, an `answer is X` expression, and a final-sentence
fallback. It also validates option letters for multiple-choice questions.

## Operational notes

- Reusing a `--run-id` skips sample identifiers already present in
  `debate.jsonl`.
- Running `src/mllm_backend.py` directly produces single-model diagnostic
  responses under `mme_outputs/<dataset>/<run-id>/`. Paper responses were
  generated through `main.py`.
- The submitted `data/` and `runs/` directories are intentionally empty.
  `--parquet` and `--out-dir` can redirect these locations.
