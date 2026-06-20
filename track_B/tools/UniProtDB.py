from __future__ import annotations

from typing import Any, Literal

import requests


UniProtFormat = Literal["json", "tsv", "fasta", "xml", "list"]


class UniProtDB:
    """
    UniProt REST client.

    Зачем нужен:
    - identity layer: accession, gene names, aliases;
    - protein function, domains, keywords;
    - GO MF/BP/CC;
    - subcellular location;
    - cross-references to Ensembl/MGI/etc.

    В Bio Property Graph это главный источник node attributes для Protein/Gene.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.base_url = "https://rest.uniprot.org"
        self.timeout = timeout

    def search(
        self,
        query: str,
        fields: list[str],
        size: int = 25,
        reviewed: bool | None = None,
        fmt: UniProtFormat = "json",
    ) -> Any:
        """Generic UniProtKB search with explicit returned fields."""
        if reviewed is not None:
            query = f"({query}) AND reviewed:{str(reviewed).lower()}"

        response = requests.get(
            f"{self.base_url}/uniprotkb/search",
            params={
                "query": query,
                "fields": ",".join(fields),
                "size": size,
                "format": fmt,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json() if fmt == "json" else response.text

    def get_entry(self, accession: str, fmt: UniProtFormat = "json") -> Any:
        """Full UniProtKB entry by accession."""
        response = requests.get(
            f"{self.base_url}/uniprotkb/{accession}",
            params={"format": fmt},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json() if fmt == "json" else response.text

    def get_gene_mouse(self, gene_symbol: str, size: int = 5) -> Any:
        """Mouse-specific UniProt lookup for one gene symbol."""
        return self.search(
            query=f"gene_exact:{gene_symbol} AND organism_id:10090",
            fields=[
                "accession",
                "id",
                "gene_names",
                "protein_name",
                "organism_id",
                "go",
                "go_f",
                "go_p",
                "go_c",
                "cc_function",
                "cc_subcellular_location",
                "keyword",
                "xref_ensembl",
                "xref_mgi",
            ],
            size=size,
            fmt="json",
        )

    def id_mapping_run(self, from_db: str, to_db: str, ids: list[str]) -> dict[str, Any]:
        """Start UniProt ID mapping job."""
        response = requests.post(
            f"{self.base_url}/idmapping/run",
            data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def id_mapping_status(self, job_id: str) -> dict[str, Any]:
        """Check UniProt ID mapping job status."""
        response = requests.get(
            f"{self.base_url}/idmapping/status/{job_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def id_mapping_results(self, job_id: str, fmt: UniProtFormat = "json") -> Any:
        """Fetch UniProt ID mapping results."""
        response = requests.get(
            f"{self.base_url}/idmapping/results/{job_id}",
            params={"format": fmt},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json() if fmt == "json" else response.text
