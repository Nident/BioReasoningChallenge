import requests
import pandas as pd
from io import StringIO


STRING_API_URL = "https://string-db.org/api"
DEFAULT_SPECIES = 10090  # Mus musculus / mouse


class GeneInfo:
    def __init__(
        self,
        species: int = DEFAULT_SPECIES,
        caller_identity: str = "mlgenx_gene_info",
        timeout: int = 30,
    ):
        self.species = species
        self.caller_identity = caller_identity
        self.timeout = timeout

    def _get_tsv(self, endpoint: str, params: dict) -> pd.DataFrame:
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
        Maps gene symbols to STRING IDs.

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
        Gets functional annotation for mapped STRING IDs.
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

    def get_enrichment(self, string_ids: list[str]) -> pd.DataFrame:
        """
        Gets GO / KEGG / Reactome / Pfam / etc. enrichment for gene set.
        For only 1-2 genes, enrichment may be sparse, but still useful.
        """
        if not string_ids:
            return pd.DataFrame()

        return self._get_tsv(
            "enrichment",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
            },
        )

    def get_interaction_partners(
        self,
        string_id: str,
        limit: int = 10,
        required_score: int = 400,
    ) -> pd.DataFrame:
        """
        Gets top interaction partners for one protein.
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
        Gets STRING network edges around provided proteins.
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
    def _df_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict]:
        if df is None or df.empty:
            return []

        if max_rows is not None:
            df = df.head(max_rows)

        return df.where(pd.notnull(df), None).to_dict(orient="records")

    def get_pair_gene_info(
        self,
        pert: str,
        target: str,
        partner_limit: int = 10,
        required_score: int = 400,
        enrichment_max_rows: int = 20,
    ) -> dict:
        """
        Main function.

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

        mapping_by_query = {}

        for _, row in mapped_df.iterrows():
            query = row.get("queryItem")
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
        enrichment_df = self.get_enrichment(string_ids)

        partners = {}

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

            partners[gene] = self._df_records(partners_df, max_rows=partner_limit)

        return {
            "status": "ok" if not missing else "partial_mapping",
            "pert": pert,
            "target": target,
            "missing_genes": missing,
            "mapping": mapping_by_query,
            "functional_annotation": self._df_records(annotation_df),
            "enrichment": self._df_records(enrichment_df, max_rows=enrichment_max_rows),
            "interaction_partners": partners,
        }
    


if __name__ == "__main__":
    import json

    client = GeneInfo()

    result = client.get_pair_gene_info("Slc35b1", "Pdia6")

    print(json.dumps(result, indent=2))