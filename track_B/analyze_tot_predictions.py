from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "train_tot_predictions_200.csv"
DEFAULT_LABELS = ("up", "down", "none")


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def numeric_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
            "sum": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p95": float(values.quantile(0.95)),
        "sum": float(values.sum()),
    }


def value_counts_frame(series: pd.Series, name: str) -> pd.DataFrame:
    frame = series.astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    frame["fraction"] = frame["count"] / frame["count"].sum()
    return frame


def build_label_metrics(ok_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        true_mask = ok_df["true_label"].astype(str) == label
        pred_mask = ok_df["pred_label"].astype(str) == label
        true_positive = int((true_mask & pred_mask).sum())
        false_positive = int((~true_mask & pred_mask).sum())
        false_negative = int((true_mask & ~pred_mask).sum())
        support = int(true_mask.sum())
        predicted = int(pred_mask.sum())
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def build_confidence_bins(ok_df: pd.DataFrame) -> pd.DataFrame:
    frame = ok_df.copy()
    frame["pred_confidence"] = pd.to_numeric(frame["pred_confidence"], errors="coerce")
    frame = frame.dropna(subset=["pred_confidence"])
    if len(frame) == 0:
        return pd.DataFrame(columns=["confidence_bin", "count", "accuracy"])

    frame["confidence_bin"] = pd.cut(
        frame["pred_confidence"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001],
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        include_lowest=True,
    )
    grouped = frame.groupby("confidence_bin", observed=False)
    result = grouped.apply(
        lambda item: pd.Series(
            {
                "count": int(len(item)),
                "accuracy": safe_div(
                    int(item["pred_label"].astype(str).eq(item["true_label"].astype(str)).sum()),
                    int(len(item)),
                ),
                "mean_confidence": float(item["pred_confidence"].mean()),
            }
        )
    ).reset_index()
    return result


def build_summary(
    df: pd.DataFrame,
    ok_df: pd.DataFrame,
    error_df: pd.DataFrame,
    label_metrics: pd.DataFrame,
    input_path: Path,
) -> dict[str, Any]:
    correct = ok_df["pred_label"].astype(str).eq(ok_df["true_label"].astype(str))
    support_sum = int(label_metrics["support"].sum()) if len(label_metrics) else 0

    weighted_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    if support_sum > 0:
        weighted_f1 = float((label_metrics["f1"] * label_metrics["support"]).sum() / support_sum)
        weighted_precision = float((label_metrics["precision"] * label_metrics["support"]).sum() / support_sum)
        weighted_recall = float((label_metrics["recall"] * label_metrics["support"]).sum() / support_sum)

    return {
        "input": str(input_path),
        "rows_total": int(len(df)),
        "ok_rows": int(len(ok_df)),
        "error_rows": int(len(error_df)),
        "coverage": safe_div(len(ok_df), len(df)),
        "accuracy": safe_div(int(correct.sum()), len(ok_df)),
        "macro_precision": float(label_metrics["precision"].mean()) if len(label_metrics) else 0.0,
        "macro_recall": float(label_metrics["recall"].mean()) if len(label_metrics) else 0.0,
        "macro_f1": float(label_metrics["f1"].mean()) if len(label_metrics) else 0.0,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "status_counts": df["status"].astype(str).value_counts(dropna=False).to_dict(),
        "true_label_counts": df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "pred_label_counts_ok": ok_df["pred_label"].astype(str).value_counts(dropna=False).to_dict(),
        "seconds": numeric_stats(df["seconds"]) if "seconds" in df.columns else {},
        "confidence_ok": numeric_stats(ok_df["pred_confidence"]) if "pred_confidence" in ok_df.columns else {},
    }


def save_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    status_counts: pd.DataFrame,
    true_counts: pd.DataFrame,
    pred_counts: pd.DataFrame,
    label_metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    confidence_bins: pd.DataFrame,
    wrong_df: pd.DataFrame,
    error_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status_counts.to_csv(output_dir / "status_counts.csv", index=False)
    true_counts.to_csv(output_dir / "true_label_counts.csv", index=False)
    pred_counts.to_csv(output_dir / "pred_label_counts.csv", index=False)
    label_metrics.to_csv(output_dir / "label_metrics.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    confidence_bins.to_csv(output_dir / "confidence_bins.csv", index=False)
    wrong_df.to_csv(output_dir / "wrong_predictions.csv", index=False)
    error_df.to_csv(output_dir / "error_rows.csv", index=False)


def print_summary(summary: dict[str, Any], label_metrics: pd.DataFrame, confusion: pd.DataFrame) -> None:
    print(f"input: {summary['input']}")
    print(f"rows_total: {summary['rows_total']}")
    print(f"ok_rows: {summary['ok_rows']}")
    print(f"error_rows: {summary['error_rows']}")
    print(f"coverage: {summary['coverage']:.4f}")
    print(f"accuracy: {summary['accuracy']:.4f}")
    print(f"macro_f1: {summary['macro_f1']:.4f}")
    print(f"weighted_f1: {summary['weighted_f1']:.4f}")
    print("\nlabel_metrics:")
    print(label_metrics.to_string(index=False))
    print("\nconfusion_matrix:")
    print(confusion.to_string())


def analyze(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir) if args.output_dir else input_path.parent / f"{input_path.stem}_stats"

    df = pd.read_csv(input_path)
    required_columns = {"true_label", "pred_label", "status"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {input_path}: {sorted(missing_columns)}")

    ok_df = df[df["status"].astype(str) == "ok"].copy()
    error_df = df[df["status"].astype(str) != "ok"].copy()
    ok_df["is_correct_calc"] = ok_df["pred_label"].astype(str).eq(ok_df["true_label"].astype(str))

    labels = list(dict.fromkeys([*DEFAULT_LABELS, *ok_df["true_label"].astype(str), *ok_df["pred_label"].astype(str)]))
    label_metrics = build_label_metrics(ok_df, labels)
    confusion = pd.crosstab(
        ok_df["true_label"].astype(str),
        ok_df["pred_label"].astype(str),
        rownames=["true_label"],
        colnames=["pred_label"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    status_counts = value_counts_frame(df["status"], "status")
    true_counts = value_counts_frame(df["true_label"], "true_label")
    pred_counts = value_counts_frame(ok_df["pred_label"], "pred_label")
    confidence_bins = build_confidence_bins(ok_df)
    wrong_df = ok_df[~ok_df["is_correct_calc"]].copy()
    summary = build_summary(df, ok_df, error_df, label_metrics, input_path)

    if not args.no_save:
        save_outputs(
            output_dir=output_dir,
            summary=summary,
            status_counts=status_counts,
            true_counts=true_counts,
            pred_counts=pred_counts,
            label_metrics=label_metrics,
            confusion=confusion,
            confidence_bins=confidence_bins,
            wrong_df=wrong_df,
            error_df=error_df,
        )
        print(f"saved_stats_dir: {output_dir}")

    print_summary(summary, label_metrics, confusion)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
