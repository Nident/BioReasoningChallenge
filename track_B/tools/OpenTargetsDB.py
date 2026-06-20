from __future__ import annotations

from typing import Any

import requests


class OpenTargetsDB:
    """
    Open Targets Platform GraphQL client.

    Зачем нужен:
    - target identity;
    - known drugs for a target;
    - target-disease associations;
    - useful mainly for human target/drug prior.

    For mouse genes, use orthology before querying target-level drug evidence.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.url = "https://api.platform.opentargets.org/api/v4/graphql"
        self.timeout = timeout

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Low-level GraphQL request."""
        response = requests.post(
            self.url,
            json={"query": query, "variables": variables},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search(self, query: str, entity_names: list[str] | None = None, page_size: int = 10) -> dict[str, Any]:
        """Search entities: targets, diseases, drugs."""
        gql = """
        query Search($queryString: String!, $entityNames: [String!], $page: Pagination) {
          search(queryString: $queryString, entityNames: $entityNames, page: $page) {
            hits { id name entity }
          }
        }
        """
        return self.graphql(gql, {"queryString": query, "entityNames": entity_names, "page": {"size": page_size, "index": 0}})

    def target(self, ensembl_id: str) -> dict[str, Any]:
        """Target card by Ensembl gene ID."""
        gql = """
        query Target($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id approvedSymbol approvedName biotype genomicLocation { chromosome start end strand }
          }
        }
        """
        return self.graphql(gql, {"ensemblId": ensembl_id})

    def known_drugs(self, ensembl_id: str, size: int = 20) -> dict[str, Any]:
        """Known drugs table for target."""
        gql = """
        query KnownDrugs($ensemblId: String!, $size: Int!) {
          target(ensemblId: $ensemblId) {
            knownDrugs(size: $size) {
              rows {
                drugId prefName drugType mechanismOfAction phase status
                disease { id name }
              }
            }
          }
        }
        """
        return self.graphql(gql, {"ensemblId": ensembl_id, "size": size})

    def associated_diseases(self, ensembl_id: str, size: int = 20) -> dict[str, Any]:
        """Disease associations for target."""
        gql = """
        query Diseases($ensemblId: String!, $page: Pagination) {
          target(ensemblId: $ensemblId) {
            associatedDiseases(page: $page) {
              rows { score disease { id name } }
            }
          }
        }
        """
        return self.graphql(gql, {"ensemblId": ensembl_id, "page": {"size": size, "index": 0}})
