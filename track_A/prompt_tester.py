from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

try:
    from .model import Model
except ImportError:
    from model import Model


ROOT_DIR = Path(__file__).resolve().parents[1]


class PromptTester:
    ANSWER_TO_LABEL = {
        "<answer>a</answer>": "up",
        "<answer>b</answer>": "down",
        "<answer>c</answer>": "none",
        "up": "up",
        "down": "down",
        "none": "none",
        "no-change": "none",
    }
    LABELS = {"up", "down", "none"}

    def __init__(
        self,
        model: Model,
        cache_dir: str | Path = ".prompt_cache",
        max_samples: int | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_samples = max_samples
        self.max_retries = max_retries
        self.results: dict[str, dict] = {}

    def run(
        self,
        data: pd.DataFrame,
        run_name: str,
        prompt_column: str = "prompt",
        label_column: str | None = "label",
        id_column: str = "id",
        output_dir: str | Path | None = None,
    ) -> dict:
        self._expect_columns(data, [id_column, prompt_column])
        if label_column:
            self._expect_columns(data, [label_column])

        rows = data.head(self.max_samples) if self.max_samples else data
        if rows.empty:
            raise ValueError("data is empty")

        output_path: Path | None = None
        run_dir: Path | None = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            run_dir = output_path / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "responses.jsonl").write_text("", encoding="utf-8")

        records = []

        for _, row in rows.iterrows():
            sample_id = self._expect_str(row[id_column], id_column)
            prompt = self._expect_str(row[prompt_column], prompt_column)
            label = self._expect_label(row[label_column]) if label_column else None
            response = self._request(prompt, sample_id, run_name)
            prediction = ""
            parse_error: ValueError | None = None
            if response:
                try:
                    prediction = self._parse_response(response)
                except ValueError as error:
                    parse_error = error

            record = {
                "id": sample_id,
                "label": label,
                "prediction": prediction,
                "response": response,
                "prompt": prompt,
            }
            records.append(record)

            if output_path is not None and run_dir is not None:
                with open(run_dir / "responses.jsonl", "a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                result = {
                    "run_name": run_name,
                    "records": records,
                    "metrics": self._metrics(records),
                }
                self.results[run_name] = result
                pd.DataFrame(records).to_csv(run_dir / "predictions.csv", index=False)
                (run_dir / "metrics.json").write_text(
                    json.dumps(result["metrics"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.compare().to_csv(output_path / "comparison.csv", index=False)

            if parse_error is not None:
                raise parse_error

        result = {
            "run_name": run_name,
            "records": records,
            "metrics": self._metrics(records),
        }
        self.results[run_name] = result
        return result

    def test_prompts(self, *args, **kwargs) -> dict:
        return self.run(*args, **kwargs)

    def save_analysis(self, output_dir: str | Path = "prompt_analysis") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        comparison = self.compare()
        comparison.to_csv(output_path / "comparison.csv", index=False)

        for run_name, result in self.results.items():
            run_dir = output_path / run_name
            run_dir.mkdir(exist_ok=True)
            pd.DataFrame(result["records"]).to_csv(run_dir / "predictions.csv", index=False)
            (run_dir / "metrics.json").write_text(
                json.dumps(result["metrics"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with open(run_dir / "responses.jsonl", "w", encoding="utf-8") as file:
                for record in result["records"]:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def compare(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "run_name": name,
                    **result["metrics"],
                }
                for name, result in self.results.items()
            ]
        )

    def print_results(self, run_name: str) -> None:
        if run_name not in self.results:
            raise KeyError(f"Unknown run_name: {run_name}")

        metrics = self.results[run_name]["metrics"]
        print(f"\n{run_name}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    def _request(self, prompt: str, sample_id: str, run_name: str) -> str:
        cache_key = hashlib.md5(
            f"{run_name}|{sample_id}|{self.model.model_name}|{prompt}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            response = payload["response"]
            if not isinstance(response, str):
                raise TypeError("cached response must be a string")
            return response

        response = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.model.request(prompt)
            except ValueError as error:
                if str(error) != "model returned empty response":
                    raise
                response = ""

            if response:
                break
            time.sleep(attempt)

        cache_path.write_text(
            json.dumps(
                {"response": response, "saved_at": datetime.now().isoformat()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return response

    def _parse_response(self, response: str) -> str:
        normalized_response = response.strip().lower()
        if normalized_response not in self.ANSWER_TO_LABEL:
            raise ValueError(f"Unexpected model response: {response}")
        return self.ANSWER_TO_LABEL[normalized_response]

    def _metrics(self, records: list[dict]) -> dict:
        evaluated_records = [record for record in records if record["prediction"] in self.LABELS]
        labels = [record["label"] for record in evaluated_records]
        predictions = [record["prediction"] for record in evaluated_records]
        skipped_samples = len(records) - len(evaluated_records)

        if any(label is None for label in labels):
            return {
                "total_samples": len(records),
                "evaluated_samples": len(evaluated_records),
                "skipped_samples": skipped_samples,
                "prediction_up": predictions.count("up"),
                "prediction_down": predictions.count("down"),
                "prediction_none": predictions.count("none"),
            }
        if not evaluated_records:
            return {
                "total_samples": len(records),
                "evaluated_samples": 0,
                "skipped_samples": skipped_samples,
            }

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=["up", "down", "none"],
            average="weighted",
            zero_division=cast(Any, 0),
        )

        return {
            "total_samples": len(records),
            "evaluated_samples": len(evaluated_records),
            "skipped_samples": skipped_samples,
            "accuracy": round(sum(a == b for a, b in zip(labels, predictions)) / len(evaluated_records), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 4),
        }

    @staticmethod
    def _expect_columns(data: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    @staticmethod
    def _expect_str(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
        return value

    def _expect_label(self, value: object) -> str:
        if value not in self.LABELS:
            raise ValueError(f"Unexpected label: {value}")
        return str(value)


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

    max_samples_raw = os.getenv("TEST_MAX_SAMPLES", "").strip().lower()
    max_samples = None if max_samples_raw in {"", "none", "null"} else int(max_samples_raw)

    label_column_raw = os.getenv("TEST_LABEL_COLUMN", "label").strip()
    label_column = label_column_raw or None

    model = Model(
        model_name=_env("MODEL_NAME"),
        api_key=_env("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL", ""),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "32")),
        top_p=float(os.getenv("MODEL_TOP_P", "1")),
    )
    tester = PromptTester(
        model=model,
        cache_dir=_project_path(os.getenv("TEST_CACHE_DIR", ".prompt_cache")),
        max_samples=max_samples,
        max_retries=int(os.getenv("TEST_MAX_RETRIES", "3")),
    )

    prompts = pd.read_csv(_project_path(_env("TEST_PROMPTS_PATH")))
    run_name = _env("TEST_RUN_NAME")
    tester.run(
        data=prompts,
        run_name=run_name,
        prompt_column=os.getenv("TEST_PROMPT_COLUMN", "prompt"),
        label_column=label_column,
        id_column=os.getenv("TEST_ID_COLUMN", "id"),
        output_dir=_project_path(_env("TEST_OUTPUT_DIR")),
    )
    tester.print_results(run_name)
    tester.save_analysis(_project_path(_env("TEST_OUTPUT_DIR")))
    print(f"Saved analysis: {_project_path(_env('TEST_OUTPUT_DIR'))}")
