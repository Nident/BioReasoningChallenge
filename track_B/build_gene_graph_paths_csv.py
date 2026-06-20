from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd

try:
    from track_B.tools.GeneGraph import GeneGraph
except ModuleNotFoundError:
    from tools.GeneGraph import GeneGraph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "train_gene_graph_paths.csv"


def json_dumps(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def path_to_text(path: Any) -> str:
    if not path:
        return ""
    return " -> ".join(path)


def flatten_graph_result(row: pd.Series, result: dict[str, Any]) -> dict[str, Any]:
    best_report = result.get("best_weighted_path_report") or {}

    return {
        "id": row.get("id", ""),
        "pert": row["pert"],
        "target": row["gene"],
        "label": row.get("label", ""),
        "status": result.get("status", ""),
        "direct_edge": result.get("direct_edge"),
        "num_nodes": result.get("num_nodes"),
        "num_edges": result.get("num_edges"),
        "pert_degree": result.get("pert_degree"),
        "target_degree": result.get("target_degree"),
        "num_common_neighbors": result.get("num_common_neighbors"),
        "shortest_path": path_to_text(result.get("shortest_path")),
        "shortest_path_json": json_dumps(result.get("shortest_path")),
        "path_length": result.get("path_length"),
        "best_weighted_path": path_to_text(result.get("best_weighted_path")),
        "best_weighted_path_json": json_dumps(result.get("best_weighted_path")),
        "best_weighted_path_confidence_product": result.get(
            "best_weighted_path_confidence_product",
            0.0,
        ),
        "best_weighted_path_min_edge_score": result.get(
            "best_weighted_path_min_edge_score",
            0.0,
        ),
        "best_weighted_path_path_length": best_report.get("path_length"),
        "best_weighted_path_edge_scores": json_dumps(best_report.get("edge_scores")),
        "best_weighted_path_sum_edge_score": best_report.get("sum_edge_score"),
        "best_weighted_path_report": json_dumps(best_report),
    }


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    done = pd.read_csv(output_path, usecols=["id"])
    return set(done["id"].astype(str))


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a",
        index=False,
        header=not output_path.exists() or output_path.stat().st_size == 0,
    )


def build_csv(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit)

    done_ids = load_done_ids(output_path) if args.resume else set()
    buffer: list[dict[str, Any]] = []

    logging.info(
        "Starting GeneGraph CSV build input=%s output=%s rows=%s resume_done=%s",
        input_path,
        output_path,
        len(df),
        len(done_ids),
    )

    for index, row in df.iterrows():
        row_id = str(row["id"])
        if row_id in done_ids:
            continue

        pert = str(row["pert"])
        target = str(row["gene"])
        logging.info("Processing %s/%s id=%s pert=%s target=%s", index + 1, len(df), row_id, pert, target)

        try:
            result = GeneGraph(
                pert=pert,
                target=target,
                required_score=args.required_score,
                add_nodes=args.add_nodes,
                n_random_paths=args.n_random_paths,
                random_path_cutoff=args.random_path_cutoff,
                top_k_paths=args.top_k_paths,
                request_timeout=args.request_timeout,
                max_path_candidates=args.max_path_candidates,
            ).get_path()
            output_row = flatten_graph_result(row, result)
        except Exception as exc:
            logging.exception("Failed id=%s pert=%s target=%s", row_id, pert, target)
            output_row = {
                "id": row_id,
                "pert": pert,
                "target": target,
                "label": row.get("label", ""),
                "status": "error",
                "error": str(exc),
            }

        buffer.append(output_row)

        if len(buffer) >= args.flush_every:
            append_rows(output_path, buffer)
            logging.info("Flushed rows=%s output=%s", len(buffer), output_path)
            buffer.clear()

        if args.sleep > 0:
            sleep(args.sleep)

    append_rows(output_path, buffer)
    logging.info("Done output=%s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--required-score", type=int, default=400)
    parser.add_argument("--add-nodes", type=int, default=100)
    parser.add_argument("--n-random-paths", type=int, default=3)
    parser.add_argument("--random-path-cutoff", type=int, default=5)
    parser.add_argument("--top-k-paths", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-path-candidates", type=int, default=20000)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_csv(parse_args())
