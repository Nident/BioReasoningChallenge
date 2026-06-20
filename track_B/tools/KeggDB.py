from __future__ import annotations

import requests


class KeggDB:
    """
    KEGG REST client.

    Зачем нужен:
    - pathway membership;
    - pathway maps;
    - gene <-> pathway links;
    - KO/module/reaction links when needed.

    Mouse organism code is hardcoded as `mmu`.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.base_url = "https://rest.kegg.jp"
        self.timeout = timeout
        self.mouse = "mmu"

    def info(self, database: str) -> str:
        """Metadata for KEGG database."""
        response = requests.get(f"{self.base_url}/info/{database}", timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def list(self, database: str) -> str:
        """List entries in a KEGG database."""
        response = requests.get(f"{self.base_url}/list/{database}", timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def find(self, database: str, query: str) -> str:
        """Search KEGG database by text query."""
        response = requests.get(f"{self.base_url}/find/{database}/{query}", timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def get(self, entry_ids: list[str], option: str | None = None) -> str:
        """Get raw KEGG flat-file records."""
        path = f"{self.base_url}/get/{'+'.join(entry_ids)}"
        if option:
            path = f"{path}/{option}"
        response = requests.get(path, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def conv(self, target_db: str, source_db: str) -> str:
        """Convert IDs between KEGG and external databases."""
        response = requests.get(f"{self.base_url}/conv/{target_db}/{source_db}", timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def link(self, target_db: str, source_db: str) -> str:
        """Get KEGG cross-links between databases."""
        response = requests.get(f"{self.base_url}/link/{target_db}/{source_db}", timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def mouse_pathways(self) -> str:
        """All mouse pathways."""
        return self.list(f"pathway/{self.mouse}")

    def mouse_gene_pathways(self, kegg_gene_id: str) -> str:
        """Pathways containing one mouse KEGG gene ID, e.g. mmu:16176."""
        return self.link("pathway", kegg_gene_id)

    def mouse_pathway_genes(self, pathway_id: str) -> str:
        """Mouse genes contained in one KEGG pathway, e.g. path:mmu04620."""
        return self.link(self.mouse, pathway_id)
