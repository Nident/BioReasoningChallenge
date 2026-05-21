# BioReasoningChallenge

Utilities for building Track A prompts, running them through an OpenAI-compatible model, saving model responses, and collecting evaluation metrics.

## Project Structure

```text
BioReasoningChallenge/
├── config/
│   ├── .env.example          # Public config template
│   └── .env                  # Local config with secrets, ignored by git
├── data/
│   ├── train.csv             # Input dataset
│   └── built_prompts/        # Generated prompts grouped by prompt type
├── prompt_analysis/          # Model responses, predictions, and metrics
├── prompt_strategies/
│   └── track_a/              # Prompt templates
├── track_A/
│   ├── model.py              # Model client wrapper
│   ├── prompt_constructor.py # Builds and saves prompts
│   └── prompt_tester.py      # Runs prompts and saves analysis
└── requirements.txt
```

## Environment Setup

Create and activate a virtual environment from the project root:

```bash
cd /Users/nident/Desktop/JOB/Kaggle/BioReasoningChallenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local env file:

```bash
cp config/.env.example config/.env
```

`config/.env` is ignored by git. Put API keys and local machine paths there only.

## Config

Prompt constructor variables:

```env
DATA_PATH=data/train.csv
PROMPT_FEW_SHOT_PATH=data/few_shot_examples_from_train.csv
PROMPT_TYPE=zero_shot_strict
PROMPT_TEMPLATE_PATH=prompt_strategies/track_a/zero_shot_strict.txt
PROMPT_OUTPUT_DIR=built_prompts
PROMPT_MAX_SAMPLES=5
```

`PROMPT_MAX_SAMPLES` can be empty, `None`, or `null` to process all rows.

Model and tester variables:

```env
MODEL_NAME=openai/gpt-oss-120b:free
MODEL_BASE_URL=https://openrouter.ai/api/v1
MODEL_API_KEY=
MODEL_TEMPERATURE=0
MODEL_MAX_TOKENS=32
MODEL_TOP_P=1

TEST_PROMPTS_PATH=data/built_prompts/zero_shot_strict/prompts.csv
TEST_RUN_NAME=zero_shot_strict
TEST_OUTPUT_DIR=prompt_analysis
TEST_CACHE_DIR=.prompt_cache
TEST_MAX_SAMPLES=5
TEST_MAX_RETRIES=3
TEST_PROMPT_COLUMN=prompt
TEST_LABEL_COLUMN=label
TEST_ID_COLUMN=id
```

`TEST_MAX_SAMPLES` can be empty, `None`, or `null` to run all prompts. `TEST_MAX_RETRIES` controls retries for empty model responses. Set `TEST_LABEL_COLUMN=` if the prompt file has no labels.

## Build Prompts

Run:

```bash
python track_A/prompt_constructor.py
```

The constructor reads `DATA_PATH`, fills `PROMPT_TEMPLATE_PATH`, optionally adds few-shot examples from `PROMPT_FEW_SHOT_PATH`, and writes results to:

```text
data/<PROMPT_OUTPUT_DIR>/<PROMPT_TYPE>/prompts.csv
data/<PROMPT_OUTPUT_DIR>/<PROMPT_TYPE>/000000_<id>.txt
data/<PROMPT_OUTPUT_DIR>/<PROMPT_TYPE>/000001_<id>.txt
```

The numeric prefix prevents files from being overwritten when several rows have the same gene pair.

## Run Prompts

Add `MODEL_API_KEY` to `config/.env`, then run:

```bash
python track_A/prompt_tester.py
```

The tester reads prompts from `TEST_PROMPTS_PATH`, sends each prompt through `track_A/model.py`, caches raw model responses in `TEST_CACHE_DIR`, and saves analysis to:

```text
<TEST_OUTPUT_DIR>/comparison.csv
<TEST_OUTPUT_DIR>/<TEST_RUN_NAME>/responses.jsonl
<TEST_OUTPUT_DIR>/<TEST_RUN_NAME>/predictions.csv
<TEST_OUTPUT_DIR>/<TEST_RUN_NAME>/metrics.json
```

Expected model answer format is strict:

```text
<answer>A</answer> -> up
<answer>B</answer> -> down
<answer>C</answer> -> none
```

Any other answer is treated as an invalid response.

## Typical Workflow

1. Configure `config/.env`.
2. Build prompts with `python track_A/prompt_constructor.py`.
3. Check generated files in `data/<PROMPT_OUTPUT_DIR>/<PROMPT_TYPE>/`.
4. Add `MODEL_API_KEY`.
5. Run `python track_A/prompt_tester.py`.
6. Inspect `prompt_analysis/<TEST_RUN_NAME>/metrics.json` and `predictions.csv`.

## Type Checking

Run Pyright on the main modules:

```bash
pyright track_A/prompt_constructor.py track_A/prompt_tester.py track_A/model.py
```
