from __future__ import annotations

from io import StringIO
from typing import Literal

import pandas as pd
import requests


Organism = Literal[9606, 10090, 10116]


class OmniPathDB:
    """
    OmniPath web-service client.

    Зачем нужен:
    - directed/signed signaling interactions;
    - enzyme-substrate / PTM edges;
    - protein complexes;
    - annotations and intercellular communication roles.

    Это основной источник для causal paths:
    pert -> kinase/adaptor/receptor -> TF -> target.
    """

    def __init__(self, organism: Organism = 10090, timeout: int = 30) -> None:
        self.base_url = "https://omnipathdb.org"
        self.organism = organism
        self.timeout = timeout

    def get_tsv(self, endpoint: str, params: dict[str, object]) -> pd.DataFrame:
        """Low-level TSV request to OmniPath endpoint."""
        params["format"] = "tsv"
        params["organism"] = self.organism
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        if not response.text.strip():
            return pd.DataFrame()
        return pd.read_csv(StringIO(response.text), sep="\t")

    def interactions(
        self,
        genesymbols: bool = True,
        directed: bool = True,
        signed: bool = True,
        datasets: str = "omnipath",
    ) -> pd.DataFrame:
        """Directed/signed signaling interactions."""
        return self.get_tsv(
            "interactions",
            {
                "genesymbols": int(genesymbols),
                "directed": int(directed),
                "signed": int(signed),
                "datasets": datasets,
            },
        )

    def interactions_for_source(self, source: str) -> pd.DataFrame:
        """All directed/signed interactions outgoing from source gene/protein."""
        return self.get_tsv(
            "interactions",
            {
                "genesymbols": 1,
                "directed": 1,
                "signed": 1,
                "sources": source,
            },
        )

    def interactions_for_target(self, target: str) -> pd.DataFrame:
        """All directed/signed interactions incoming to target gene/protein."""
        return self.get_tsv(
            "interactions",
            {
                "genesymbols": 1,
                "directed": 1,
                "signed": 1,
                "targets": target,
            },
        )

    def enzsub(self, genesymbols: bool = True) -> pd.DataFrame:
        """Enzyme-substrate relationships, useful for phosphorylation/PTM paths."""
        return self.get_tsv("enzsub", {"genesymbols": int(genesymbols)})

    def complexes(self) -> pd.DataFrame:
        """Protein complex membership."""
        return self.get_tsv("complexes", {})

    def annotations(self, proteins: list[str] | None = None) -> pd.DataFrame:
        """Protein annotations from OmniPath integrated resources."""
        params: dict[str, object] = {"genesymbols": 1}
        if proteins:
            params["proteins"] = ",".join(proteins)
        return self.get_tsv("annotations", params)

    def intercell(self, proteins: list[str] | None = None) -> pd.DataFrame:
        """Ligand/receptor/secreted/transmembrane role annotations."""
        params: dict[str, object] = {"genesymbols": 1}
        if proteins:
            params["proteins"] = ",".join(proteins)
        return self.get_tsv("intercell", params)
