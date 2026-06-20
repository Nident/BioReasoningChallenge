from __future__ import annotations

from io import StringIO
from typing import Any, Literal

import pandas as pd
import requests


StringOutput = Literal["tsv", "json"]


class StringDB:
    """
    STRING API client.

    Зачем нужен:
    - functional/physical protein association graph;
    - быстрые paths между pert и target;
    - evidence channels: experimental, database, coexpression, textmining, etc.

    Важно:
    STRING score не означает activation/repression и не дает up/down.
    Это confidence, что белки функционально связаны.
    """

    def __init__(
        self,
        species: int = 10090,
        caller_identity: str = "mlgenx_stringdb",
        timeout: float | tuple[float, float] = (5.0, 15.0),
    ) -> None:
        self.base_url = "https://string-db.org/api"
        self.species = species
        self.caller_identity = caller_identity
        self.timeout = timeout

    def get_tsv(self, endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
        """Low-level TSV request к STRING endpoint."""
        params["caller_identity"] = self.caller_identity
        response = requests.get(
            f"{self.base_url}/tsv/{endpoint}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        if not response.text.strip():
            return pd.DataFrame()
        return pd.read_csv(StringIO(response.text), sep="\t")

    def get_json(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Low-level JSON request к STRING endpoint."""
        params["caller_identity"] = self.caller_identity
        response = requests.get(
            f"{self.base_url}/json/{endpoint}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_string_ids(self, genes: list[str], limit: int = 1) -> pd.DataFrame:
        """Map gene symbols to STRING protein IDs."""
        return self.get_tsv(
            "get_string_ids",
            {
                "identifiers": "\r".join(genes),
                "species": self.species,
                "limit": limit,
                "echo_query": 1,
            },
        )

    def network(
        self,
        string_ids: list[str],
        required_score: int = 400,
        add_nodes: int = 20,
        network_type: Literal["functional", "physical"] = "functional",
    ) -> pd.DataFrame:
        """STRING network around input proteins plus optional add_nodes."""
        return self.get_tsv(
            "network",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
                "required_score": required_score,
                "add_nodes": add_nodes,
                "network_type": network_type,
            },
        )

    def interaction_partners(
        self,
        string_id: str,
        required_score: int = 400,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Top STRING partners for one mapped STRING protein ID."""
        return self.get_tsv(
            "interaction_partners",
            {
                "identifiers": string_id,
                "species": self.species,
                "required_score": required_score,
                "limit": limit,
            },
        )

    def functional_annotation(self, string_ids: list[str]) -> pd.DataFrame:
        """Functional annotations for STRING proteins."""
        return self.get_tsv(
            "functional_annotation",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
                "allow_pubmed": 1,
            },
        )

    def enrichment(self, string_ids: list[str]) -> pd.DataFrame:
        """GO/pathway/domain enrichment for a set of STRING proteins."""
        return self.get_tsv(
            "enrichment",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
            },
        )

    def ppi_enrichment(self, string_ids: list[str]) -> pd.DataFrame:
        """PPI enrichment statistics for a protein set."""
        return self.get_tsv(
            "ppi_enrichment",
            {
                "identifiers": "\r".join(string_ids),
                "species": self.species,
            },
        )
