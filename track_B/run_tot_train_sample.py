from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import pandas as pd

try:
    from track_B.TreeOfT import TreeOfThought
    from track_B.tot_steps import DEFAULT_PAIR_EVIDENCE_TOOLS
except ModuleNotFoundError:
    from TreeOfT import TreeOfThought
    from tot_steps import DEFAULT_PAIR_EVIDENCE_TOOLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "train_tot_predictions_200.csv"
DEFAULT_DEBUG_ROOT = PROJECT_ROOT / "track_B" / "debug_tot_200"
DEFAULT_QUESTION = "What is the effect of perturbation gene {pert_gene} on gene {target_gene}?"

OUTPUT_COLUMNS = [
    "sample_position",
    "train_index",
    "id",
    "pert",
    "target",
    "true_label",
    "pred_answer",
    "pred_label",
    "pred_confidence",
    "pred_hypothesis",
    "pred_id",
    "pred_parent_id",
    "is_correct",
    "status",
    "error",
    "seconds",
    "debug_dir",
]

EVIDENCE_TOOL_PRESETS = {
    "default": DEFAULT_PAIR_EVIDENCE_TOOLS,
    "core": (
        "gene_pair_info",
        "gene_graph_paths",
        "uniprotdb_evidence",
        "trrustdb_evidence",
    ),
    "minimal": (
        "gene_pair_info",
        "uniprotdb_evidence",
    ),
}


def safe_path_part(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)[:160]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def parse_evidence_tools(value: str) -> tuple[str, ...]:
    if value in EVIDENCE_TOOL_PRESETS:
        return EVIDENCE_TOOL_PRESETS[value]
    return tuple(item.strip() for item in value.split(",") if item.strip())


def select_rows(df: pd.DataFrame, sample_size: int, seed: int, use_head: bool) -> pd.DataFrame:
    if sample_size > len(df):
        raise ValueError(f"sample_size={sample_size} is larger than dataset rows={len(df)}")

    if use_head:
        selected = df.head(sample_size).copy()
    else:
        selected = df.sample(n=sample_size, random_state=seed).copy()

    return selected.reset_index(names="train_index")


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    done = pd.read_csv(output_path, usecols=["id", "status"])
    done = done[done["status"].astype(str).isin({"ok", "error"})]
    return set(done["id"].astype(str))


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=OUTPUT_COLUMNS)
    frame.to_csv(
        output_path,
        mode="a",
        index=False,
        header=not output_path.exists() or output_path.stat().st_size == 0,
    )


def build_result_row(
    sample_position: int,
    row: pd.Series,
    result: Any,
    seconds: float,
    debug_dir: str,
) -> dict[str, Any]:
    result_dict = asdict(result)
    true_label = str(row.get("label", ""))
    pred_label = str(result_dict["label"])

    return {
        "sample_position": sample_position,
        "train_index": int(row["train_index"]),
        "id": row["id"],
        "pert": row["pert"],
        "target": row["gene"],
        "true_label": true_label,
        "pred_answer": result_dict["answer"],
        "pred_label": pred_label,
        "pred_confidence": result_dict["confidence"],
        "pred_hypothesis": result_dict["hypothesis"],
        "pred_id": result_dict["id"],
        "pred_parent_id": result_dict["parent_id"],
        "is_correct": pred_label == true_label,
        "status": "ok",
        "error": "",
        "seconds": round(seconds, 3),
        "debug_dir": debug_dir,
    }


def build_error_row(
    sample_position: int,
    row: pd.Series,
    error: Exception,
    seconds: float,
    debug_dir: str,
) -> dict[str, Any]:
    return {
        "sample_position": sample_position,
        "train_index": int(row["train_index"]),
        "id": row["id"],
        "pert": row["pert"],
        "target": row["gene"],
        "true_label": row.get("label", ""),
        "pred_answer": "",
        "pred_label": "",
        "pred_confidence": "",
        "pred_hypothesis": "",
        "pred_id": "",
        "pred_parent_id": "",
        "is_correct": "",
        "status": "error",
        "error": str(error),
        "seconds": round(seconds, 3),
        "debug_dir": debug_dir,
    }


def prepare_output(output_path: Path, resume: bool, overwrite: bool) -> None:
    if not output_path.exists():
        return

    if resume:
        return

    if overwrite:
        output_path.unlink()
        return

    raise FileExistsError(f"{output_path} already exists. Use --resume or --overwrite.")


def run_batch(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    debug_root = resolve_path(args.debug_root) if args.debug_root else None
    evidence_tools = parse_evidence_tools(args.evidence_tools)

    df = pd.read_csv(input_path)
    selected = select_rows(
        df=df,
        sample_size=args.sample_size,
        seed=args.seed,
        use_head=args.head,
    )

    if args.dry_run:
        print(selected[["train_index", "id", "pert", "gene", "label"]].to_string(index=False))
        print(f"rows={len(selected)} output={output_path}")
        print(f"evidence_tools={','.join(evidence_tools)}")
        return

    prepare_output(output_path, resume=args.resume, overwrite=args.overwrite)
    done_ids = load_done_ids(output_path) if args.resume else set()

    logging.info(
        "Starting ToT batch input=%s output=%s rows=%s done=%s max_depth=%s evidence_tools=%s",
        input_path,
        output_path,
        len(selected),
        len(done_ids),
        args.max_depth,
        ",".join(evidence_tools),
    )

    tot = TreeOfThought()
    buffer: list[dict[str, Any]] = []

    for sample_position, row in selected.iterrows():
        row_id = str(row["id"])
        if row_id in done_ids:
            continue

        if debug_root is not None:
            debug_dir = debug_root / f"{sample_position:04d}_{safe_path_part(row_id)}"
            os.environ["TOT_DEBUG_DIR"] = str(debug_dir)
        else:
            debug_dir = Path(os.getenv("TOT_DEBUG_DIR", PROJECT_ROOT / "track_B" / "debug"))

        started = perf_counter()
        logging.info(
            "Processing %s/%s id=%s pert=%s target=%s label=%s debug=%s",
            sample_position + 1,
            len(selected),
            row_id,
            row["pert"],
            row["gene"],
            row.get("label", ""),
            debug_dir,
        )

        try:
            result = tot.recursive_tot(
                question=args.question,
                pert_gene=str(row["pert"]),
                target_gene=str(row["gene"]),
                max_depth=args.max_depth,
                max_hypotheses_per_step=args.max_hypotheses_per_step,
                new_hypotheses_per_survivor=args.new_hypotheses_per_survivor,
                min_score=args.min_score,
                max_memory_items=args.max_memory_items,
                evidence_tool_names=evidence_tools,
            )
            seconds = perf_counter() - started
            output_row = build_result_row(
                sample_position=sample_position,
                row=row,
                result=result,
                seconds=seconds,
                debug_dir=str(debug_dir),
            )
            logging.info(
                "Done id=%s pred=%s true=%s confidence=%s seconds=%.1f",
                row_id,
                output_row["pred_label"],
                output_row["true_label"],
                output_row["pred_confidence"],
                seconds,
            )
        except Exception as exc:
            seconds = perf_counter() - started
            logging.exception("Failed id=%s pert=%s target=%s", row_id, row["pert"], row["gene"])
            output_row = build_error_row(
                sample_position=sample_position,
                row=row,
                error=exc,
                seconds=seconds,
                debug_dir=str(debug_dir),
            )

        buffer.append(output_row)
        done_ids.add(row_id)

        if len(buffer) >= args.flush_every:
            append_rows(output_path, buffer)
            logging.info("Flushed rows=%s output=%s", len(buffer), output_path)
            buffer.clear()

        if args.sleep > 0:
            sleep(args.sleep)

    append_rows(output_path, buffer)

    result_df = pd.read_csv(output_path)
    ok_df = result_df[result_df["status"] == "ok"].copy()
    if len(ok_df) > 0:
        accuracy = ok_df["pred_label"].astype(str).eq(ok_df["true_label"].astype(str)).mean()
        logging.info("Finished output=%s ok_rows=%s accuracy=%.4f", output_path, len(ok_df), accuracy)
    else:
        logging.info("Finished output=%s ok_rows=0", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-hypotheses-per-step", type=int, default=5)
    parser.add_argument("--new-hypotheses-per-survivor", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--max-memory-items", type=int, default=100)
    parser.add_argument("--evidence-tools", default="core")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_batch(parse_args())
