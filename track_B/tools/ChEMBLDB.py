from __future__ import annotations

from typing import Any

import requests


class ChEMBLDB:
    """
    ChEMBL API client.

    Зачем нужен:
    - слой drugs / compounds / bioactivity;
    - поиск лекарств и молекул, которые таргетят белок;
    - mechanism of action, target binding, assay activity, drug indications.

    В Bio Property Graph это дает ребра:
    Drug -> TARGETS -> Protein
    Drug -> HAS_MECHANISM -> Mechanism
    Drug -> TESTED_IN_ASSAY -> Target
    """

    def __init__(self, timeout: int = 30) -> None:
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.timeout = timeout

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Low-level GET для любого ChEMBL endpoint."""
        response = requests.get(
            f"{self.base_url}/{endpoint}.json",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def molecule(self, chembl_id: str) -> dict[str, Any]:
        """Детальная карточка молекулы по ChEMBL ID, например CHEMBL25."""
        return self.get(f"molecule/{chembl_id}")

    def target(self, target_chembl_id: str) -> dict[str, Any]:
        """Детальная карточка drug target по ChEMBL target ID."""
        return self.get(f"target/{target_chembl_id}")

    def target_search(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Поиск ChEMBL targets по gene/protein имени."""
        return self.get("target/search", {"q": query, "limit": limit})

    def molecule_search(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Поиск молекул/лекарств по имени."""
        return self.get("molecule/search", {"q": query, "limit": limit})

    def mechanism_by_molecule(self, molecule_chembl_id: str) -> dict[str, Any]:
        """Mechanism of action для молекулы."""
        return self.get("mechanism", {"molecule_chembl_id": molecule_chembl_id})

    def mechanism_by_target(self, target_chembl_id: str) -> dict[str, Any]:
        """Все known mechanisms, где указанный target является MoA target."""
        return self.get("mechanism", {"target_chembl_id": target_chembl_id})

    def activity_by_target(self, target_chembl_id: str, limit: int = 100) -> dict[str, Any]:
        """Bioactivity assays для target: IC50/Ki/EC50/etc."""
        return self.get("activity", {"target_chembl_id": target_chembl_id, "limit": limit})

    def activity_by_molecule(self, molecule_chembl_id: str, limit: int = 100) -> dict[str, Any]:
        """Bioactivity assays для молекулы против разных targets."""
        return self.get("activity", {"molecule_chembl_id": molecule_chembl_id, "limit": limit})

    def drug_indication(self, molecule_chembl_id: str) -> dict[str, Any]:
        """Clinical indications для drug molecule."""
        return self.get("drug_indication", {"molecule_chembl_id": molecule_chembl_id})
