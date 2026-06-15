from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field
from langchain.tools import tool



PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class GenePairInput(BaseModel):
    pert: str = Field(description="Perturbation gene symbol, for example Slc35b1")
    gene: str = Field(description="Target gene symbol, for example Pdia6")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, str]
    runner: Callable[[dict[str, Any]], Any]


def _gene_pair_info(args: dict[str, Any]) -> dict[str, Any]:
    data = {}
    from track_B.tools.GeneInfo import GeneInfo

    gene_info_data = GeneInfo().get_pair_gene_info(
        pert=str(args["pert"]),
        target=str(args["gene"]),
    )

    data["status"] = gene_info_data.get("status")
    data["pert"] = gene_info_data.get("pert")
    data["gene"] = gene_info_data.get("target")
    data["missing_genes"] = gene_info_data.get("missing_genes", [])
    data["compact_gene_summary"] = gene_info_data.get("compact_gene_summary", {})
    data["compact_pair_report"] = gene_info_data.get("compact_pair_report", {})

    return data


def _gene_graph_paths(args: dict[str, Any]) -> dict[str, Any]:
    data = {}
    from track_B.tools.GeneGraph import GeneGraph

    gene_graph_data = GeneGraph(
        pert=str(args["pert"]),
        target=str(args["gene"]),
        required_score=400,
        add_nodes=20,
        n_random_paths=3,
        random_path_cutoff=5,
        top_k_paths=4
    ).get_path()

    data["status"] = gene_graph_data.get("status")
    data["direct_edge"] = gene_graph_data.get("direct_edge")
    data["weighted_paths"] = gene_graph_data.get("best_weighted_path")
    data["shortest_path"] = gene_graph_data.get("shortest_path")
    data["common_neighbors"] = gene_graph_data.get("common_neighbors", [])
    data["top_paths"] = gene_graph_data.get("top_paths", [])

    return data


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "gene_pair_info": ToolSpec(
        name="gene_pair_info",
        description=(
            "Returns compact STRING functional annotations, classes, shared pathways, "
            "shared biological processes, and interaction partners for a gene pair."
        ),
        args_schema={
            "pert": "Perturbation gene symbol, for example Slc35b1",
            "gene": "Target gene symbol, for example Pdia6",
        },
        runner=_gene_pair_info,
    ),
    "gene_graph_paths": ToolSpec(
        name="gene_graph_paths",
        description=(
            "Returns STRING graph connectivity evidence: direct edge, common neighbors, "
            "shortest paths, weighted paths, and edge evidence channels."
        ),
        args_schema={
            "pert": "Perturbation gene symbol, for example Slc35b1",
            "gene": "Target gene symbol, for example Pdia6",
        },
        runner=_gene_graph_paths,
    ),
}


class LangChainTools:
    @staticmethod
    def tool_descriptions() -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": spec.args_schema,
            }
            for spec in TOOL_REGISTRY.values()
        ]

    @staticmethod
    def tool_descriptions_json() -> str:
        return _to_json(LangChainTools.tool_descriptions())

    @staticmethod
    def run_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": f"Unknown tool: {tool_name}",
                "result": None,
            }

        try:
            return {
                "tool_name": tool_name,
                "ok": True,
                "args": args,
                "result": spec.runner(args),
            }
        except Exception as exc:
            return {
                "tool_name": tool_name,
                "ok": False,
                "args": args,
                "error": str(exc),
                "result": None,
            }

    @staticmethod
    def collect_tool_results(pert: str, gene: str) -> list[dict[str, Any]]:
        args = {"pert": pert, "gene": gene}
        return [
            LangChainTools.run_tool("gene_pair_info", args),
            LangChainTools.run_tool("gene_graph_paths", args),
        ]

    @staticmethod
    def collect_tool_evidence(pert: str, gene: str) -> str:
        return _to_json(LangChainTools.collect_tool_results(pert=pert, gene=gene))

    @staticmethod
    def build_gene_tools() -> list[Any]:
        @tool("gene_pair_info", args_schema=GenePairInput)
        def gene_pair_info(pert: str, gene: str) -> str:
            """Return compact STRING functional annotations for a perturbation-target gene pair."""
            return _to_json(_gene_pair_info({"pert": pert, "gene": gene}))

        @tool("gene_graph_paths", args_schema=GenePairInput)
        def gene_graph_paths(pert: str, gene: str) -> str:
            """Return STRING graph connectivity, paths, common neighbors, and edge evidence."""
            return _to_json(_gene_graph_paths({"pert": pert, "gene": gene}))

        return [gene_pair_info, 
                gene_graph_paths]


def tool_descriptions() -> list[dict[str, Any]]:
    return LangChainTools.tool_descriptions()


def tool_descriptions_json() -> str:
    return LangChainTools.tool_descriptions_json()


def run_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return LangChainTools.run_tool(tool_name=tool_name, args=args)


def collect_tool_results(pert: str, gene: str) -> list[dict[str, Any]]:
    return LangChainTools.collect_tool_results(pert=pert, gene=gene)


def collect_tool_evidence(pert: str, gene: str) -> str:
    return LangChainTools.collect_tool_evidence(pert=pert, gene=gene)


def build_gene_tools() -> list[Any]:
    return LangChainTools.build_gene_tools()


def _to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, indent=2)


def main() -> None:
   
    tool = 'list'
    pert = 'Slc35b1'
    gene = 'Pdia6'
    if tool == "list":
        print(tool_descriptions_json())
        return

    if tool == "all":
        print(_to_json(collect_tool_results(pert=pert, gene=gene)))
        return

    print(_to_json(run_tool(tool, {"pert": pert, "gene": gene})))


if __name__ == "__main__":
    main()
