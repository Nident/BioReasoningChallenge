import requests
import pandas as pd
from io import StringIO
from collections.abc import Sequence
from typing import Any


STRING_API_URL = "https://string-db.org/api"
DEFAULT_SPECIES = 10090  # Mus musculus / mouse

GENERIC_TERMS = {
    "biological process",
    "cellular process",
    "metabolic process",
    "cellular metabolic process",
    "primary metabolic process",
    "organic substance metabolic process",
    "intracellular",
    "cell",
    "cell part",
    "cellular anatomical entity",
    "binding",
    "protein binding",
}

STRING_INTERACTION_SCORE_ANNOTATIONS = {
    "score": "Combined STRING confidence score for this functional association.",
    "nscore": "Neighborhood evidence score based on genomic neighborhood.",
    "fscore": "Gene fusion evidence score based on observed fused genes in other genomes.",
    "pscore": "Phylogenetic co-occurrence evidence score based on genes appearing or disappearing together across genomes.",
    "ascore": "Co-expression evidence score based on correlated expression patterns.",
    "escore": "Experimental evidence score based on direct or indirect experimental data.",
    "dscore": "Database evidence score based on curated pathway or interaction databases.",
    "tscore": "Text-mining evidence score based on co-mentions in scientific text.",
}


class GeneInfo:
    def __init__(
        self,
        species: int = DEFAULT_SPECIES,
        caller_identity: str = "mlgenx_gene_info",
        timeout: int = 30,
    ):
        """Configure STRING API access for one species and caller identity."""
        self.species = species
        self.caller_identity = caller_identity
        self.timeout = timeout

    def _get_tsv(self, endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
        """
        Call a STRING `/tsv/{endpoint}` API endpoint and parse the TSV response.

        This is the low-level HTTP helper used by all STRING-backed methods below.
        It returns an empty DataFrame when STRING returns an empty body.
        """
        params = dict(params)
        params["caller_identity"] = self.caller_identity

        url = f"{STRING_API_URL}/tsv/{endpoint}"
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        text = response.text.strip()

        if not text:
            return pd.DataFrame()

        return pd.read_csv(StringIO(text), sep="\t")

    def map_genes(self, genes: list[str]) -> pd.DataFrame:
        """
        Fetch STRING identifiers for input gene symbols.

        STRING endpoint:
            get_string_ids

        Retrieves the best STRING mapping for each query gene, including the STRING
        protein ID, preferred name, taxon ID, and STRING's compact gene annotation.

        Returns columns usually like:
        queryItem, stringId, preferredName, annotation, ncbiTaxonId, ...
        """
        genes = list(dict.fromkeys(genes))  # remove duplicates, keep order

        return self._get_tsv(
            "get_string_ids",
            {
                "identifiers": "\r".join(genes),
                "species": self.species,
                "limit": 1,
                "echo_query": 1,
            },
        )

    def get_gene_annotation(self, string_ids: list[str]) -> pd.DataFrame:
        """
        Fetch functional annotations for mapped STRING protein IDs.

        STRING endpoint:
            functional_annotation

        Retrieves category-level annotations such as function descriptions,
        biological processes, pathways, compartments, keywords, domains, and
        related annotation sources for each STRING protein ID.
        """
        if not string_ids:
            return pd.DataFrame()

        return self._get_tsv(
            "functional_annotation",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
                "allow_pubmed": 0,
            },
        )

        # def get_enrichment(self, string_ids: list[str]) -> pd.DataFrame:
        #     """
        #     Fetch enrichment terms for a set of STRING protein IDs.
        #
        #     STRING endpoint:
        #         enrichment
        #
        #     Retrieves GO / KEGG / Reactome / Pfam / etc. enrichment for a gene set.
        #     For only 1-2 genes, enrichment may be sparse, but still useful.
        #     """
        #     if not string_ids:
        #         return pd.DataFrame()

        #     return self._get_tsv(
        #         "enrichment",
        #         {
        #             "identifiers": "\r".join(string_ids),
        #             "species": self.species,
        #         },
        #     )

    def get_interaction_partners(
        self,
        string_id: str,
        limit: int = 10,
        required_score: int = 400,
    ) -> pd.DataFrame:
        """
        Fetch top STRING interaction partners for one mapped protein.

        STRING endpoint:
            interaction_partners

        Retrieves high-scoring functional association partners for one STRING
        protein ID, filtered by `required_score` and capped by `limit`.

        Important returned score fields:
            score: combined STRING confidence score.
            nscore: genomic neighborhood evidence.
            fscore: gene fusion evidence.
            pscore: phylogenetic co-occurrence evidence.
            ascore: co-expression evidence.
            escore: experimental evidence.
            dscore: curated database evidence.
            tscore: text-mining evidence.
        """
        return self._get_tsv(
            "interaction_partners",
            {
                "identifiers": string_id,
                "species": self.species,
                "limit": limit,
                "required_score": required_score,
            },
        )

    def get_network(
        self,
        string_ids: list[str],
        required_score: int = 400,
        add_nodes: int = 20,
    ) -> pd.DataFrame:
        """
        Fetch a STRING functional network around the provided proteins.

        STRING endpoint:
            network

        Retrieves association edges among the provided STRING IDs plus optional
        neighboring nodes. Returned columns usually include the two protein names,
        combined score, and evidence-channel scores.
        """
        if not string_ids:
            return pd.DataFrame()

        return self._get_tsv(
            "network",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
                "required_score": required_score,
                "add_nodes": add_nodes,
                "network_type": "functional",
            },
        )

    @staticmethod
    def _df_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
        """Convert a STRING response DataFrame into JSON-serializable records."""
        if df is None or df.empty:
            return []

        if max_rows is not None:
            df = df.head(max_rows)

        records = df.where(pd.notnull(df), None).to_dict(orient="records")

        return [dict(record) for record in records]

    @staticmethod
    def _annotate_interaction_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Attach English explanations to STRING interaction score fields.

        The original API score fields stay unchanged. The additional
        `score_annotations` object repeats each available API score together with
        a short explanation so saved JSON files are self-describing.
        """
        annotated_records = []

        for record in records:
            annotated_record = dict(record)
            annotated_record["score_annotations"] = {
                score_name: {
                    "api_score": record.get(score_name),
                    "description": description,
                }
                for score_name, description in STRING_INTERACTION_SCORE_ANNOTATIONS.items()
                if score_name in record
            }
            annotated_records.append(annotated_record)

        return annotated_records

    @staticmethod
    def _unique_values(values: Sequence[Any], max_items: int = 8) -> list[str]:
        """Deduplicate annotation values and remove overly generic STRING terms."""
        result: list[str] = []

        for value in values:
            if not value:
                continue

            value = str(value).strip()
            if not value:
                continue

            if value.lower() in GENERIC_TERMS:
                continue

            if value not in result:
                result.append(value)

            if len(result) >= max_items:
                break

        return result

    @staticmethod
    def _infer_gene_class(text: str) -> str:
        """Infer a coarse gene class from STRING annotation text."""
        text = text.lower()

        if "transcription factor" in text or "dna-binding transcription" in text:
            return "transcription regulator"
        if "transporter" in text or "solute carrier" in text:
            return "transporter"
        if "kinase" in text:
            return "kinase/signaling enzyme"
        if "phosphatase" in text:
            return "phosphatase/signaling enzyme"
        if "receptor" in text:
            return "receptor"
        if "protein folding" in text or "chaperone" in text or "disulfide" in text:
            return "protein folding / chaperone related"
        if "ribosomal" in text or "ribosome" in text:
            return "ribosomal / translation related"
        if "mitochond" in text:
            return "mitochondrial related"
        if "immune" in text or "cytokine" in text or "interferon" in text:
            return "immune / inflammatory related"

        return "not clearly classified"

    def _gene_records(
        self,
        annotation_df: pd.DataFrame,
        gene: str,
    ) -> list[dict[str, Any]]:
        """Select functional annotation rows that belong to one preferred gene name."""
        if annotation_df is None or annotation_df.empty:
            return []
        if "preferredNames" not in annotation_df.columns:
            return []

        records = self._df_records(annotation_df)
        gene_records: list[dict[str, Any]] = []

        for record in records:
            names = str(record.get("preferredNames") or "")
            preferred_names = [item.strip() for item in names.split(",")]

            if gene in preferred_names:
                gene_records.append(record)

        return gene_records

    def _compact_gene_summary(
        self,
        gene: str,
        mapping: dict[str, Any],
        annotation_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Build a compact per-gene summary from STRING mapping and annotation data.

        The summary keeps the most useful fields for prompting: STRING ID,
        description, inferred gene class, functions, GO biological processes,
        pathways, compartments, and keywords.
        """
        records = self._gene_records(annotation_df, gene)

        def descriptions_for(categories: Sequence[str], max_items: int = 8) -> list[str]:
            if not records:
                return []

            descriptions = [
                str(record.get("description") or "")
                for record in records
                if record.get("category") in categories
            ]

            return self._unique_values(descriptions, max_items=max_items)

        main_function = descriptions_for(["Function"], max_items=5)
        specific_go_bp = descriptions_for(["Process"], max_items=8)
        pathways = descriptions_for(["KEGG", "RCTM"], max_items=8)
        compartments = descriptions_for(["Component", "COMPARTMENTS"], max_items=8)
        keywords = descriptions_for(["Keyword", "InterPro", "Pfam"], max_items=8)

        text_for_class = " ".join([
            str(mapping.get("annotation") or ""),
            " ".join(main_function),
            " ".join(specific_go_bp),
            " ".join(pathways),
            " ".join(keywords),
        ])

        return {
            "gene": gene,
            "preferred_name": mapping.get("preferred_name"),
            "string_id": mapping.get("string_id"),
            "description": mapping.get("annotation"),
            "gene_class": self._infer_gene_class(text_for_class),
            "main_function": main_function,
            "specific_go_bp": specific_go_bp,
            "pathways": pathways,
            "compartments": compartments,
            "keywords": keywords,
        }

    def _compact_pair_report(
        self,
        pert: str,
        target: str,
        summaries: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compare two compact gene summaries and report shared STRING annotations.

        This does not call STRING directly. It derives shared pathways, biological
        processes, and compartments from previously fetched annotation rows.
        """
        pert_summary = summaries.get(pert, {})
        target_summary = summaries.get(target, {})

        shared_processes = sorted(set(
            pert_summary.get("specific_go_bp", [])
        ) & set(target_summary.get("specific_go_bp", [])))
        shared_pathways = sorted(set(
            pert_summary.get("pathways", [])
        ) & set(target_summary.get("pathways", [])))
        shared_compartments = sorted(set(
            pert_summary.get("compartments", [])
        ) & set(target_summary.get("compartments", [])))

        lines = [
            f"Perturbation gene {pert}: {pert_summary.get('gene_class', 'unknown')}.",
            f"Target gene {target}: {target_summary.get('gene_class', 'unknown')}.",
        ]

        if shared_pathways:
            lines.append("Shared pathways: " + "; ".join(shared_pathways[:5]) + ".")
        if shared_processes:
            lines.append("Shared GO biological processes: " + "; ".join(shared_processes[:5]) + ".")
        if shared_compartments:
            lines.append("Shared compartments: " + "; ".join(shared_compartments[:5]) + ".")
        if not shared_pathways and not shared_processes and not shared_compartments:
            lines.append("No clear shared pathway/process/compartment in compact STRING annotations.")

        return {
            "shared_processes": shared_processes,
            "shared_pathways": shared_pathways,
            "shared_compartments": shared_compartments,
            "report": " ".join(lines),
        }

    def get_pair_gene_info(
        self,
        pert: str,
        target: str,
        partner_limit: int = 10,
        required_score: int = 400,
        enrichment_max_rows: int = 20,
    ) -> dict[str, Any]:
        """
        Fetch and assemble STRING evidence for a perturbation-target gene pair.

        STRING endpoints used:
            get_string_ids
            functional_annotation
            interaction_partners

        This method maps both genes to STRING IDs, fetches their functional
        annotations, fetches top interaction partners for each mapped protein,
        and builds compact per-gene and pair-level summaries for downstream LLM
        prompts.

        Input:
            pert: perturbation gene symbol, e.g. "Slc35b1"
            target: target gene symbol, e.g. "Pdia6"

        Output:
            dict with mapping, descriptions, annotations, enrichment,
            and top STRING partners for both genes.
        """
        genes = [pert, target]

        mapped_df = self.map_genes(genes)

        if mapped_df.empty:
            return {
                "status": "mapping_failed",
                "pert": pert,
                "target": target,
                "error": "No STRING mappings found.",
            }

        mapping_by_query: dict[str, dict[str, Any]] = {}

        for _, row in mapped_df.iterrows():
            query_value = row.get("queryItem")
            if query_value is None:
                continue

            query = str(query_value)
            if query not in mapping_by_query:
                mapping_by_query[query] = {
                    "query": query,
                    "string_id": row.get("stringId"),
                    "preferred_name": row.get("preferredName"),
                    "annotation": row.get("annotation"),
                    "ncbi_taxon_id": row.get("ncbiTaxonId"),
                }

        missing = [g for g in genes if g not in mapping_by_query]

        string_ids = [
            mapping_by_query[g]["string_id"]
            for g in genes
            if g in mapping_by_query and mapping_by_query[g]["string_id"] is not None
        ]

        annotation_df = self.get_gene_annotation(string_ids)
        # enrichment_df = self.get_enrichment(string_ids)

        # print("Enrichment DataFrame:")
        # annotation_df.to_csv("annotation_df.tsv", sep="\t", index=False)

        partners: dict[str, list[dict[str, Any]]] = {}
        compact_gene_summary: dict[str, dict[str, Any]] = {}

        for gene in genes:
            if gene not in mapping_by_query:
                partners[gene] = []
                continue

            sid = mapping_by_query[gene]["string_id"]
            partners_df = self.get_interaction_partners(
                sid,
                limit=partner_limit,
                required_score=required_score,
            )

            # partners_df.to_csv(f"{gene}_partners_df.tsv", sep="\t", index=False)
            partner_records = self._df_records(partners_df, max_rows=partner_limit)
            partners[gene] = self._annotate_interaction_scores(partner_records)
            compact_gene_summary[gene] = self._compact_gene_summary(
                gene,
                mapping_by_query[gene],
                annotation_df,
            )

        compact_pair_report = self._compact_pair_report(
            pert,
            target,
            compact_gene_summary,
        )

        return {
            "status": "ok" if not missing else "partial_mapping",
            "pert": pert,
            "target": target,
            "missing_genes": missing,
            "mapping": mapping_by_query,
            "compact_gene_summary": compact_gene_summary,
            "compact_pair_report": compact_pair_report,
            "functional_annotation": self._df_records(annotation_df),
            # "enrichment": self._df_records(enrichment_df, max_rows=enrichment_max_rows),
            "interaction_score_legend": STRING_INTERACTION_SCORE_ANNOTATIONS,
            "interaction_partners": partners,
        }
    


if __name__ == "__main__":
    import json

    client = GeneInfo()

    result = client.get_pair_gene_info("Slc35b1", "Pdia6")

    with open("track_B/debug/gene_info.json", "w") as f:
        json.dump(result, f, indent=2)
