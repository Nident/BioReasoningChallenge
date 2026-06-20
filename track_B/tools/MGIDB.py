from __future__ import annotations

from typing import Any

import requests


class MGIDB:
    """
    MGI/MouseMine client.

    Зачем нужен:
    - mouse-native gene identity;
    - mouse GO annotations;
    - mouse phenotype annotations;
    - good source for Mus musculus-specific evidence.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.base_url = "https://www.mousemine.org/mousemine/service"
        self.timeout = timeout

    def query(self, xml_query: str, fmt: str = "jsonobjects") -> Any:
        """Low-level MouseMine XML query."""
        response = requests.get(
            f"{self.base_url}/query/results",
            params={"query": xml_query, "format": fmt},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json() if fmt == "jsonobjects" else response.text

    def gene_by_symbol(self, symbol: str) -> Any:
        """Mouse gene identity by official symbol."""
        xml = f"""
        <query model="genomic" view="Gene.primaryIdentifier Gene.symbol Gene.name Gene.organism.name Gene.sequenceOntologyTerm.name" sortOrder="Gene.symbol ASC">
          <constraint path="Gene.symbol" op="=" value="{symbol}"/>
          <constraint path="Gene.organism.name" op="=" value="Mus musculus"/>
        </query>
        """
        return self.query(xml)

    def gene_go_annotations(self, symbol: str) -> Any:
        """Mouse GO annotations for gene symbol."""
        xml = f"""
        <query model="genomic" view="Gene.symbol Gene.goAnnotation.ontologyTerm.identifier Gene.goAnnotation.ontologyTerm.name Gene.goAnnotation.evidence.code" sortOrder="Gene.symbol ASC">
          <constraint path="Gene.symbol" op="=" value="{symbol}"/>
          <constraint path="Gene.organism.name" op="=" value="Mus musculus"/>
        </query>
        """
        return self.query(xml)

    def gene_phenotypes(self, symbol: str) -> Any:
        """Mouse phenotype annotations linked to gene alleles."""
        xml = f"""
        <query model="genomic" view="Gene.symbol Gene.alleles.phenotypeAnnotations.ontologyTerm.identifier Gene.alleles.phenotypeAnnotations.ontologyTerm.name" sortOrder="Gene.symbol ASC">
          <constraint path="Gene.symbol" op="=" value="{symbol}"/>
          <constraint path="Gene.organism.name" op="=" value="Mus musculus"/>
        </query>
        """
        return self.query(xml)
