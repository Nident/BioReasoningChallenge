# Track A Prompt Strategy Pack

This folder contains prompt templates for the BioReasoning Challenge Track A.
All templates use only `{pert}` and `{gene}` placeholders.

Recommended escalation order:

1. `00_zero_shot_strict.txt` - minimal strict baseline.
2. `01_zero_shot_de_first_metric_aware.txt` - zero-shot with metric-aware decision order.
3. `02_few_shot_balanced_from_train.txt` - balanced few-shot examples from `data/train.csv`.
4. `03_few_shot_biology_calibrated.txt` - few-shot plus biological evidence rules.
5. `04_combined_de_first_fewshot_strict.txt` - combined strategy, recommended first serious run.
6. `05_finetune_inference_prompt.txt` - use after zero-shot and few-shot underperform, mainly for Track C inference after SFT/LoRA.

Suggested command:

```bash
uv run python examples/track_a_logprobs.py \
  --api-base http://localhost:8000/v1 \
  --prompt-template prompt_strategies/track_a/04_combined_de_first_fewshot_strict.txt \
  --output-dir outputs/track_a/combined_de_first_fewshot
```

Why logprobs are preferred:

- The Kaggle score is AUROC-based, so ranked probabilities matter.
- `track_a_logprobs.py` extracts continuous `prediction_up` and `prediction_down`
  from the A/B/C answer-token probabilities.
- The strict `<answer>X</answer>` format makes extraction reliable.

Few-shot examples are listed in `few_shot_examples_from_train.csv`. They are
real rows from `data/train.csv`, selected to balance `up`, `down`, and `none`
and to calibrate the model toward choosing `C` when significant evidence is weak.
