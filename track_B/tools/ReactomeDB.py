from __future__ import annotations

from typing import Any

import requests


class ReactomeDB:
    """
    Reactome Content/Analysis Service client.

    Зачем нужен:
    - curated pathway and reaction layer;
    - pathway membership for genes/proteins;
    - event participants;
    - pathway enrichment / projection.

    Useful for paths like pert -> pathway -> downstream process -> target.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.base_url = "https://reactome.org/ContentService"
        self.analysis_url = "https://reactome.org/AnalysisService"
        self.timeout = timeout

    def search(self, query: str, species: str = "Mus musculus") -> dict[str, Any]:
        """Search Reactome entities/events by text."""
        response = requests.get(
            f"{self.base_url}/search/query",
            params={"query": query, "species": species},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def data_query(self, stable_id: str) -> dict[str, Any]:
        """Fetch one Reactome entity/event by stable ID."""
        response = requests.get(f"{self.base_url}/data/query/{stable_id}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def pathways_low_entity(self, identifier: str, species: str = "Mus musculus") -> list[dict[str, Any]]:
        """Pathways containing a low-level entity identifier."""
        response = requests.get(
            f"{self.base_url}/data/pathways/low/entity/{identifier}",
            params={"species": species},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def pathways_top_level(self, species: str = "Mus musculus") -> list[dict[str, Any]]:
        """Top-level Reactome pathways for species."""
        response = requests.get(
            f"{self.base_url}/data/pathways/top/{species}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def participants(self, stable_id: str) -> list[dict[str, Any]]:
        """Participants of a Reactome event/pathway."""
        response = requests.get(
            f"{self.base_url}/data/participants/{stable_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def contained_events(self, stable_id: str) -> list[dict[str, Any]]:
        """Nested events under one Reactome pathway/event."""
        response = requests.get(
            f"{self.base_url}/data/event/{stable_id}/containedEvents",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def analyze_identifiers(self, identifiers: list[str], species: str = "Mus musculus") -> dict[str, Any]:
        """Reactome pathway analysis for a list of identifiers."""
        response = requests.post(
            f"{self.analysis_url}/identifiers/projection",
            params={"species": species},
            data="\n".join(identifiers),
            headers={"Content-Type": "text/plain"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
