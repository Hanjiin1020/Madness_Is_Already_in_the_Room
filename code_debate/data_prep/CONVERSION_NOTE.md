# Benchmark conversion notes

The four converters produce the same parquet interface for `main.py`:

| Column | Meaning |
|---|---|
| `index` | Stable question identifier |
| `question` | Question text without `<image...>` placeholders |
| `image` | Image bytes or a base64 string |
| `answer` | Gold option letter, binary answer, or free-form answer |
| `answer_type` | `mcq`, `binary`, or `open` prompt selector |
| `category` | Benchmark-native top-level category |
| `A`–`L` | Optional multiple-choice alternatives |

Every converter checks unique IDs, non-empty questions and images, contiguous
choice letters, valid MCQ gold letters, image decoding, and parquet round-trip
integrity. Output paths are relative to `code_debate/`; source paths are always
provided by the caller. No machine-specific path is embedded in the code.

## MMStar

Source: Hugging Face `Lin-Chen/MMStar`, validation split. The source contains
1,500 questions with alternatives embedded in the question text. The converter:

- separates inline `Options:` or `Choices:` blocks into `A`–`L` columns;
- removes the duplicated option block from `question`;
- restores non-trailing alternatives converted from the literal `None` to a
  missing value by pandas;
- preserves source category metadata and image bytes;
- verifies option continuity and that the gold letter points to a real option.

The paper retains 1,355 questions. The converter applies the shipped ID list by
default when the supplementary layout is kept; `--ids` can select another list.

```bash
python data_prep/convert_mmstar.py \
  --src /path/to/mmstar.parquet \
  --out data/MMStar_converted.parquet
```

If `--src` is omitted, `convert_mmstar.py` searches the Hugging Face cache.

## HallusionBench

Source: the public HallusionBench question table with one image per row. The
paper uses the 781 identifiers in
`data_results_analysis/clean_question_ids_hallusionbench.json`.
`convert_hallusionbench.py` applies this list by default and accepts
parquet, JSON, JSONL, or CSV input and:

- normalizes `Yes`/`No` gold answers;
- keeps the native VD/VS label in `category`;
- keeps the content category in `category_detail`;
- resolves an inline image, image dictionary, or path relative to the source
  file.

```bash
python data_prep/convert_hallusionbench.py \
  --src /path/to/HallusionBench.parquet \
  --out data/HallusionBench_converted.parquet
```

The converted `answer_type=binary` field selects the yes/no format automatically.

## MMMU

Source: public MMMU parquet shards or a merged MMMU parquet. Only rows with one
available image are retained. `convert_mmmu.py`:

- accepts either `image` or the official `image_1`–`image_7` columns;
- parses the source `options` list into `A`–`L` columns;
- preserves subject, subfield, and split metadata;
- samples an equal number per subject when no ID file is supplied.

The paper uses the 1,500 identifiers in
`data_results_analysis/clean_question_ids_mmmu.json`. The converter applies this
list by default. A caller-supplied `--ids` list takes precedence. If no ID list
is available, `--per-subject` and `--seed` provide a reproducible balanced
fallback.

```bash
python data_prep/convert_mmmu.py \
  --src /path/to/MMMU/*/*.parquet \
  --per-subject 50 --seed 42 \
  --out data/MMMU_converted.parquet
```

## MME-CoT

Source: the public single-image MME-CoT parquet files. The paper combines the
available source splits and retains the 575 identifiers in
`data_results_analysis/clean_question_ids_mmecot.json`. The converter:

- merges any number of input parquet files;
- applies the shipped ID list by default when the supplementary layout is kept;
- preserves MCQ alternatives and leaves free-form rows without alternatives;
- keeps category, subcategory, and perception/reasoning metadata.

```bash
python data_prep/convert_mme-cot.py \
  --src /path/to/MME-CoT_single_opt.parquet \
        /path/to/MME-CoT_single_test.parquet \
  --out data/MME-CoT_converted.parquet
```

The output contains both MCQ and open-ended questions. `main.py` chooses the
answer format from the presence of option columns.
