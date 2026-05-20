from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


PromptType = Literal[
    "zero_shot_strict",
    "zero_shot_de_first_metric_aware",
    "few_shot_balanced_from_train",
    "few_shot_biology_calibrated",
    "combined_de_first_fewshot_strict",
    "finetune_inference_prompt",
]


class PromptConstructor:
    PROMPT_TYPES = {
        "zero_shot_strict",
        "zero_shot_de_first_metric_aware",
        "few_shot_balanced_from_train",
        "few_shot_biology_calibrated",
        "combined_de_first_fewshot_strict",
        "finetune_inference_prompt",
    }
    LABEL_TO_ANSWER = {
        "up": "<answer>A</answer>",
        "down": "<answer>B</answer>",
        "none": "<answer>C</answer>",
    }

    def build_prompt(
        self,
        template: str,
        prompt_id: str,
        pert: str,
        gene: str,
        label: str | None = None,
        few_shot_examples: str = "",
    ) -> dict[str, str]:
        if not isinstance(template, str) or not template:
            raise TypeError("template must be a non-empty string")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise TypeError("prompt_id must be a non-empty string")
        if not isinstance(pert, str) or not pert:
            raise TypeError("pert must be a non-empty string")
        if not isinstance(gene, str) or not gene:
            raise TypeError("gene must be a non-empty string")
        if label is not None and label not in self.LABEL_TO_ANSWER:
            raise ValueError(f"Unexpected label: {label}")

        prompt_values = {
            "id": prompt_id,
            "pert": pert,
            "gene": gene,
            "label": label or "",
            "few_shot_examples": few_shot_examples,
        }
        prompt = template.format(**prompt_values)

        row = {
            "id": prompt_id,
            "pert": pert,
            "gene": gene,
        }
        if label is not None:
            row["label"] = label
        row["prompt"] = prompt
        return row

    def build_prompts(
        self,
        data_path: str | Path,
        template_path: str | Path,
        output_dir: str | Path,
        prompt_type: PromptType,
        few_shot_path: str | Path | None = None,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        if prompt_type not in self.PROMPT_TYPES:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        data = pd.read_csv(data_path)
        if data.empty:
            raise ValueError("data is empty")
        for column in ["id", "pert", "gene"]:
            if column not in data.columns:
                raise ValueError(f"Missing column: {column}")
        if max_samples is not None and max_samples > 0:
            data = data.head(max_samples)

        template = _project_path(template_path).read_text(encoding="utf-8")
        few_shot_examples = ""
        if few_shot_path is not None:
            few_shot_data = pd.read_csv(few_shot_path)
            for column in ["pert", "gene", "label"]:
                if column not in few_shot_data.columns:
                    raise ValueError(f"Missing few-shot column: {column}")
            if few_shot_data.empty:
                raise ValueError("few_shot_data is empty")

            examples: list[str] = []
            few_shot_rows = cast(list[dict[str, object]], few_shot_data.to_dict(orient="records"))
            for idx, row in enumerate(few_shot_rows, start=1):
                pert_value = row["pert"]
                gene_value = row["gene"]
                label_value = row["label"]
                reason_value = row.get("reason_for_selection", "")

                if not isinstance(pert_value, str) or not pert_value:
                    raise TypeError("few_shot.pert must be a non-empty string")
                if not isinstance(gene_value, str) or not gene_value:
                    raise TypeError("few_shot.gene must be a non-empty string")
                if not isinstance(label_value, str) or label_value not in self.LABEL_TO_ANSWER:
                    raise ValueError(f"Unexpected few-shot label: {label_value}")

                example = (
                    f"Example {idx}\n"
                    f"Perturbed gene: {pert_value}\n"
                    f"Target gene: {gene_value}\n"
                    f"Known training label: {label_value}\n"
                    f"Correct answer tag: {self.LABEL_TO_ANSWER[label_value]}"
                )
                if isinstance(reason_value, str) and reason_value:
                    example += f"\nCalibration note: {reason_value}"
                examples.append(example)

            few_shot_examples = "\n\n".join(examples)
            if "{few_shot_examples}" not in template:
                section = "\n\n### Few-shot calibration examples\n{few_shot_examples}\n"
                for marker in ("### Constraints", "### Output Format", "Current input:", "Final answer:"):
                    if marker in template:
                        template = template.replace(marker, f"{section}\n{marker}", 1)
                        break
                else:
                    template = template.rstrip() + section

        rows: list[dict[str, str]] = []
        data_rows = cast(list[dict[str, object]], data.to_dict(orient="records"))
        for row in data_rows:
            prompt_id_value = row["id"]
            pert_value = row["pert"]
            gene_value = row["gene"]
            label_value = row.get("label")

            if not isinstance(prompt_id_value, str) or not prompt_id_value:
                raise TypeError("id must be a non-empty string")
            if not isinstance(pert_value, str) or not pert_value:
                raise TypeError("pert must be a non-empty string")
            if not isinstance(gene_value, str) or not gene_value:
                raise TypeError("gene must be a non-empty string")

            label: str | None = None
            if label_value is not None:
                if not isinstance(label_value, str):
                    raise TypeError("label must be a string")
                label = label_value

            rows.append(
                self.build_prompt(
                    template=template,
                    prompt_id=prompt_id_value,
                    pert=pert_value,
                    gene=gene_value,
                    label=label,
                    few_shot_examples=few_shot_examples,
                )
            )

        prompts = pd.DataFrame(rows)
        output_path = Path(output_dir)
        if output_path.is_absolute() or ".." in output_path.parts:
            raise ValueError("output_dir must be a relative folder inside data")
        output_path = DATA_DIR / output_path
        txt_dir = output_path / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        for old_prompt in txt_dir.glob("*.txt"):
            old_prompt.unlink()

        prompts.to_csv(output_path / "prompts.csv", index=False)
        prompt_rows = cast(list[dict[str, object]], prompts.to_dict(orient="records"))
        for idx, row in enumerate(prompt_rows):
            prompt_id_value = row["id"]
            prompt_value = row["prompt"]

            if not isinstance(prompt_id_value, str) or not prompt_id_value:
                raise TypeError("id must be a non-empty string")
            if not isinstance(prompt_value, str) or not prompt_value:
                raise TypeError("prompt must be a non-empty string")
            if "/" in prompt_id_value or "\\" in prompt_id_value:
                raise ValueError(f"id is not a valid filename: {prompt_id_value}")

            (txt_dir / f"{idx:06d}_{prompt_id_value}.txt").write_text(prompt_value, encoding="utf-8")

        return prompts


def _env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / "config" / ".env")

    max_samples_raw = os.getenv("PROMPT_MAX_SAMPLES", "")
    max_samples = int(max_samples_raw) if max_samples_raw != "" else None

    constructor = PromptConstructor()
    prompts = constructor.build_prompts(
        data_path=_env("DATA_PATH"),
        template_path=_env("PROMPT_TEMPLATE_PATH"),
        output_dir=_env("PROMPT_OUTPUT_DIR"),
        prompt_type=cast(PromptType, _env("PROMPT_TYPE")),
        few_shot_path=os.getenv("PROMPT_FEW_SHOT_PATH") or None,
        max_samples=max_samples,
    )
    print(f"Saved prompts: {DATA_DIR / _env('PROMPT_OUTPUT_DIR')}")
    print(f"Total prompts: {len(prompts)}")
