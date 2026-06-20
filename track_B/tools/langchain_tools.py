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
    data["weighted_paths"] = gene_graph_data.get("weighted_paths", [])
    data["best_weighted_path"] = gene_graph_data.get("best_weighted_path")
    data["best_weighted_path_confidence_product"] = gene_graph_data.get(
        "best_weighted_path_confidence_product",
        0.0,
    )
    data["best_weighted_path_min_edge_score"] = gene_graph_data.get(
        "best_weighted_path_min_edge_score",
        0.0,
    )
    data["best_weighted_path_report"] = gene_graph_data.get(
        "best_weighted_path_report",
    )
    data["shortest_path"] = gene_graph_data.get("shortest_path")
    data["common_neighbors"] = gene_graph_data.get("common_neighbors", [])
    data["top_paths"] = gene_graph_data.get("top_paths", [])

    return data


def _stringdb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.StringDB import StringDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = StringDB(timeout=(3.0, 8.0))
    mapped_payload = _safe_value(lambda: db.get_string_ids([pert, gene]))
    if mapped_payload["error"] is not None:
        return {
            "source": "STRING",
            "pert": pert,
            "gene": gene,
            "status": "error",
            "error": mapped_payload["error"],
            "mapped": [],
            "functional_annotation": [],
            "network_edges": [],
            "ppi_enrichment": [],
        }

    mapped = mapped_payload["value"]
    string_ids = mapped["stringId"].dropna().astype(str).tolist() if "stringId" in mapped else []
    functional_annotation = _safe_value(lambda: db.functional_annotation(string_ids)) if string_ids else {"value": None, "error": None}
    network_edges = _safe_value(lambda: db.network(string_ids, add_nodes=20)) if string_ids else {"value": None, "error": None}
    ppi_enrichment = _safe_value(lambda: db.ppi_enrichment(string_ids)) if string_ids else {"value": None, "error": None}

    return {
        "source": "STRING",
        "pert": pert,
        "gene": gene,
        "status": "ok",
        "mapped": _df_records(mapped),
        "functional_annotation": _df_records(functional_annotation["value"], 12) if functional_annotation["error"] is None else [],
        "network_edges": _df_records(network_edges["value"], 25) if network_edges["error"] is None else [],
        "ppi_enrichment": _df_records(ppi_enrichment["value"], 5) if ppi_enrichment["error"] is None else [],
        "warnings": [
            warning
            for warning in [
                functional_annotation["error"],
                network_edges["error"],
                ppi_enrichment["error"],
            ]
            if warning is not None
        ],
    }


def _uniprotdb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.UniProtDB import UniProtDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = UniProtDB()

    return {
        "source": "UniProt",
        "pert": pert,
        "gene": gene,
        "pert_records": _uniprot_records(db.get_gene_mouse(pert, size=3)),
        "gene_records": _uniprot_records(db.get_gene_mouse(gene, size=3)),
    }


def _keggdb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.KeggDB import KeggDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = KeggDB()
    pert_find = db.find("mmu", pert)
    gene_find = db.find("mmu", gene)
    pert_ids = _first_kegg_ids(pert_find, 3)
    gene_ids = _first_kegg_ids(gene_find, 3)

    return {
        "source": "KEGG",
        "pert": pert,
        "gene": gene,
        "pert_find": pert_find,
        "gene_find": gene_find,
        "pert_pathways": {kegg_id: db.mouse_gene_pathways(kegg_id) for kegg_id in pert_ids},
        "gene_pathways": {kegg_id: db.mouse_gene_pathways(kegg_id) for kegg_id in gene_ids},
    }


def _reactomedb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.ReactomeDB import ReactomeDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = ReactomeDB()

    return {
        "source": "Reactome",
        "pert": pert,
        "gene": gene,
        "pert_search": _limit_json(db.search(pert), 20),
        "gene_search": _limit_json(db.search(gene), 20),
    }


def _omnipathdb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.OmniPathDB import OmniPathDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = OmniPathDB()
    outgoing = db.interactions_for_source(pert)
    incoming = db.interactions_for_target(gene)

    return {
        "source": "OmniPath",
        "pert": pert,
        "gene": gene,
        "pert_outgoing_signed": _df_records(outgoing, 25),
        "gene_incoming_signed": _df_records(incoming, 25),
        "direct_edges": _df_records(
            outgoing[outgoing["target_genesymbol"] == gene]
            if "target_genesymbol" in outgoing
            else outgoing.head(0),
            10,
        ),
    }


def _trrustdb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.TRRUSTDB import TRRUSTDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = TRRUSTDB(species="mouse", timeout=(3.0, 5.0))
    raw_payload = _safe_value(db.raw)
    if raw_payload["error"] is not None:
        return {
            "source": "TRRUST",
            "species": "mouse",
            "pert": pert,
            "gene": gene,
            "status": "error",
            "error": raw_payload["error"],
            "pair": [],
            "pert_as_tf": [],
            "gene_regulators": [],
        }

    raw = raw_payload["value"]

    return {
        "source": "TRRUST",
        "status": "ok",
        "species": "mouse",
        "pert": pert,
        "gene": gene,
        "pair": _df_records(raw[(raw["tf"] == pert) & (raw["target"] == gene)], 20),
        "pert_as_tf": _df_records(raw[raw["tf"] == pert], 25),
        "gene_regulators": _df_records(raw[raw["target"] == gene], 25),
    }


def _chipatlasdb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.ChIPAtlasDB import ChIPAtlasDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = ChIPAtlasDB()
    antigens = db.antigens(genome="mm10", ag_class="TFs and others", cl_class="Blood")

    return {
        "source": "ChIP-Atlas",
        "pert": pert,
        "gene": gene,
        "genome": "mm10",
        "cell_class": "Blood",
        "matching_antigens": _matching_json_items(antigens, pert, 20),
        "target_gene_analysis_index": _limit_json(db.target_genes_index(), 20),
        "note": "Binding evidence requires a separate target-gene/peak lookup; this tool reports available mouse Blood TF antigen data.",
    }


def _chembl_db_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.ChEMBLDB import ChEMBLDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = ChEMBLDB()

    return {
        "source": "ChEMBL",
        "pert": pert,
        "gene": gene,
        "pert_target_search": _limit_json(db.target_search(pert, limit=10), 20),
        "gene_target_search": _limit_json(db.target_search(gene, limit=10), 20),
        "note": "Mostly human/drug prior. Use mouse-human orthology before strong claims.",
    }


def _opentargets_db_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.OpenTargetsDB import OpenTargetsDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = OpenTargetsDB()

    return {
        "source": "OpenTargets",
        "pert": pert,
        "gene": gene,
        "pert_search": _limit_json(db.search(pert, page_size=5), 20),
        "gene_search": _limit_json(db.search(gene, page_size=5), 20),
        "note": "Human target/drug/disease prior. Use orthology for mouse genes.",
    }


def _geneontologydb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.GeneOntologyDB import GeneOntologyDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = GeneOntologyDB()
    pert_search = _safe_value(lambda: db.search(pert, rows=10))
    gene_search = _safe_value(lambda: db.search(gene, rows=10))

    return {
        "source": "GeneOntology",
        "pert": pert,
        "gene": gene,
        "pert_search": _limit_json(pert_search["value"], 20) if pert_search["error"] is None else {},
        "gene_search": _limit_json(gene_search["value"], 20) if gene_search["error"] is None else {},
        "warnings": [
            warning
            for warning in [pert_search["error"], gene_search["error"]]
            if warning is not None
        ],
        "note": "GO term search only. For mouse gene GO annotations prefer mgi_mouse_evidence or uniprotdb_evidence.",
    }


def _mgidb_evidence(args: dict[str, Any]) -> dict[str, Any]:
    from track_B.tools.MGIDB import MGIDB

    pert = str(args["pert"])
    gene = str(args["gene"])
    db = MGIDB()
    pert_gene = _safe_value(lambda: db.gene_by_symbol(pert))
    gene_gene = _safe_value(lambda: db.gene_by_symbol(gene))
    pert_go = _safe_value(lambda: db.gene_go_annotations(pert))
    gene_go = _safe_value(lambda: db.gene_go_annotations(gene))

    return {
        "source": "MGI/MouseMine",
        "pert": pert,
        "gene": gene,
        "pert_gene": _limit_json(pert_gene["value"], 20) if pert_gene["error"] is None else {},
        "gene_gene": _limit_json(gene_gene["value"], 20) if gene_gene["error"] is None else {},
        "pert_go": _limit_json(pert_go["value"], 20) if pert_go["error"] is None else {},
        "gene_go": _limit_json(gene_go["value"], 20) if gene_go["error"] is None else {},
        "warnings": [
            warning
            for warning in [
                pert_gene["error"],
                gene_gene["error"],
                pert_go["error"],
                gene_go["error"],
            ]
            if warning is not None
        ],
    }


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
    "stringdb_evidence": ToolSpec(
        name="stringdb_evidence",
        description="Returns STRING mappings, annotations, local network edges, and PPI enrichment for a gene pair.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_stringdb_evidence,
    ),
    "uniprotdb_evidence": ToolSpec(
        name="uniprotdb_evidence",
        description="Returns UniProt mouse protein function, GO, keywords, localization, and cross-reference records for both genes.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_uniprotdb_evidence,
    ),
    "keggdb_evidence": ToolSpec(
        name="keggdb_evidence",
        description="Returns KEGG mouse gene search records and pathway links for both genes.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_keggdb_evidence,
    ),
    "reactomedb_evidence": ToolSpec(
        name="reactomedb_evidence",
        description="Returns Reactome mouse pathway/entity search evidence for both genes.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_reactomedb_evidence,
    ),
    "omnipathdb_evidence": ToolSpec(
        name="omnipathdb_evidence",
        description="Returns OmniPath directed/signed signaling edges outgoing from pert and incoming to target.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_omnipathdb_evidence,
    ),
    "trrustdb_evidence": ToolSpec(
        name="trrustdb_evidence",
        description="Returns TRRUST mouse TF-target regulation evidence: direct pair, pert as TF, and target regulators.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_trrustdb_evidence,
    ),
    "chipatlasdb_evidence": ToolSpec(
        name="chipatlasdb_evidence",
        description="Returns ChIP-Atlas mouse availability evidence for TF/chromatin binding context.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_chipatlasdb_evidence,
    ),
    "chembl_db_evidence": ToolSpec(
        name="chembl_db_evidence",
        description="Returns ChEMBL drug-target search evidence for both genes; mostly human/drug prior.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_chembl_db_evidence,
    ),
    "opentargets_db_evidence": ToolSpec(
        name="opentargets_db_evidence",
        description="Returns Open Targets search evidence for target/drug/disease prior; mostly human and orthology-dependent.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_opentargets_db_evidence,
    ),
    "geneontologydb_evidence": ToolSpec(
        name="geneontologydb_evidence",
        description="Returns Gene Ontology search evidence for both gene names; use as GO vocabulary prior.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_geneontologydb_evidence,
    ),
    "mgidb_evidence": ToolSpec(
        name="mgidb_evidence",
        description="Returns MGI/MouseMine mouse-native gene identity and GO annotations for both genes.",
        args_schema={"pert": "Perturbation gene symbol", "gene": "Target gene symbol"},
        runner=_mgidb_evidence,
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
            LangChainTools.run_tool("uniprotdb_evidence", args),
            LangChainTools.run_tool("trrustdb_evidence", args),
            LangChainTools.run_tool("omnipathdb_evidence", args),
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

        @tool("stringdb_evidence", args_schema=GenePairInput)
        def stringdb_evidence(pert: str, gene: str) -> str:
            """Return STRING mappings, annotations, network edges, and enrichment for a gene pair."""
            return _to_json(_stringdb_evidence({"pert": pert, "gene": gene}))

        @tool("uniprotdb_evidence", args_schema=GenePairInput)
        def uniprotdb_evidence(pert: str, gene: str) -> str:
            """Return UniProt mouse function, GO, localization, keywords, and xrefs for both genes."""
            return _to_json(_uniprotdb_evidence({"pert": pert, "gene": gene}))

        @tool("keggdb_evidence", args_schema=GenePairInput)
        def keggdb_evidence(pert: str, gene: str) -> str:
            """Return KEGG mouse gene search and pathway links for both genes."""
            return _to_json(_keggdb_evidence({"pert": pert, "gene": gene}))

        @tool("reactomedb_evidence", args_schema=GenePairInput)
        def reactomedb_evidence(pert: str, gene: str) -> str:
            """Return Reactome mouse pathway/entity search evidence for both genes."""
            return _to_json(_reactomedb_evidence({"pert": pert, "gene": gene}))

        @tool("omnipathdb_evidence", args_schema=GenePairInput)
        def omnipathdb_evidence(pert: str, gene: str) -> str:
            """Return OmniPath directed/signed signaling evidence for a gene pair."""
            return _to_json(_omnipathdb_evidence({"pert": pert, "gene": gene}))

        @tool("trrustdb_evidence", args_schema=GenePairInput)
        def trrustdb_evidence(pert: str, gene: str) -> str:
            """Return TRRUST mouse TF-target regulation evidence for a gene pair."""
            return _to_json(_trrustdb_evidence({"pert": pert, "gene": gene}))

        @tool("chipatlasdb_evidence", args_schema=GenePairInput)
        def chipatlasdb_evidence(pert: str, gene: str) -> str:
            """Return ChIP-Atlas mouse TF/chromatin data availability evidence."""
            return _to_json(_chipatlasdb_evidence({"pert": pert, "gene": gene}))

        @tool("chembl_db_evidence", args_schema=GenePairInput)
        def chembl_db_evidence(pert: str, gene: str) -> str:
            """Return ChEMBL drug-target search evidence for both genes."""
            return _to_json(_chembl_db_evidence({"pert": pert, "gene": gene}))

        @tool("opentargets_db_evidence", args_schema=GenePairInput)
        def opentargets_db_evidence(pert: str, gene: str) -> str:
            """Return Open Targets search evidence for target/drug/disease prior."""
            return _to_json(_opentargets_db_evidence({"pert": pert, "gene": gene}))

        @tool("geneontologydb_evidence", args_schema=GenePairInput)
        def geneontologydb_evidence(pert: str, gene: str) -> str:
            """Return Gene Ontology search evidence for both genes."""
            return _to_json(_geneontologydb_evidence({"pert": pert, "gene": gene}))

        @tool("mgidb_evidence", args_schema=GenePairInput)
        def mgidb_evidence(pert: str, gene: str) -> str:
            """Return MGI/MouseMine mouse gene identity and GO annotations for both genes."""
            return _to_json(_mgidb_evidence({"pert": pert, "gene": gene}))

        return [
            gene_pair_info,
            gene_graph_paths,
            stringdb_evidence,
            uniprotdb_evidence,
            keggdb_evidence,
            reactomedb_evidence,
            omnipathdb_evidence,
            trrustdb_evidence,
            chipatlasdb_evidence,
            chembl_db_evidence,
            opentargets_db_evidence,
            geneontologydb_evidence,
            mgidb_evidence,
        ]


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


def _df_records(df: Any, max_rows: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    if max_rows is not None:
        df = df.head(max_rows)
    return df.where(df.notnull(), None).to_dict(orient="records")


def _uniprot_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in payload.get("results", [])[:5]:
        records.append(
            {
                "primaryAccession": item.get("primaryAccession"),
                "uniProtkbId": item.get("uniProtkbId"),
                "proteinDescription": item.get("proteinDescription"),
                "genes": item.get("genes"),
                "organism": item.get("organism"),
                "comments": item.get("comments", [])[:4],
                "keywords": item.get("keywords", [])[:20],
                "uniProtKBCrossReferences": item.get("uniProtKBCrossReferences", [])[:20],
            }
        )
    return records


def _first_kegg_ids(text: str, limit: int) -> list[str]:
    return [line.split("\t", 1)[0] for line in text.strip().splitlines()[:limit] if line.strip()]


def _limit_json(value: Any, max_items: int) -> Any:
    if isinstance(value, list):
        return value[:max_items]
    if isinstance(value, dict):
        limited = {}
        for key, item in value.items():
            if isinstance(item, list):
                limited[key] = item[:max_items]
            elif isinstance(item, dict):
                limited[key] = _limit_json(item, max_items)
            else:
                limited[key] = item
        return limited
    return value


def _matching_json_items(value: Any, text: str, max_items: int) -> list[Any]:
    matches = []
    needle = text.lower()
    items = value if isinstance(value, list) else []
    for item in items:
        if needle in json.dumps(item, ensure_ascii=False).lower():
            matches.append(item)
        if len(matches) >= max_items:
            break
    return matches


def _safe_value(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"value": fn(), "error": None}
    except Exception as exc:
        return {"value": None, "error": str(exc)}


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
