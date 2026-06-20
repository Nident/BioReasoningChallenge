from __future__ import annotations

from typing import Any

import requests


class GeneOntologyDB:
    """
    Gene Ontology API client.

    Зачем нужен:
    - controlled vocabulary for biological process, molecular function, cellular component;
    - GO term metadata;
    - gene-function annotations when GO bioentity IDs are known.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.base_url = "https://api.geneontology.org/api"
        self.quickgo_url = "https://www.ebi.ac.uk/QuickGO/services"
        self.timeout = timeout

    def ontology_term(self, go_id: str) -> dict[str, Any]:
        """GO term metadata by GO ID, e.g. GO:0006954."""
        response = requests.get(
            f"{self.base_url}/ontology/term/{go_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def bioentity_gene(self, taxon: str, gene_id: str) -> dict[str, Any]:
        """GO bioentity record for gene ID."""
        response = requests.get(
            f"{self.base_url}/bioentity/gene/{taxon}:{gene_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def gene_annotations(self, taxon: str, gene_id: str) -> dict[str, Any]:
        """GO functional annotations for gene ID."""
        response = requests.get(
            f"{self.base_url}/bioentity/gene/{taxon}:{gene_id}/function",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search(self, query: str, category: str = "ontology_class", rows: int = 20) -> dict[str, Any]:
        """Search GO terms through QuickGO."""
        response = requests.get(
            f"{self.quickgo_url}/ontology/go/search",
            params={"query": query, "limit": rows},
            headers={"Accept": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
