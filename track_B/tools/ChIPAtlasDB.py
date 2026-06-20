from __future__ import annotations

from typing import Any, Literal

import requests


AgClass = Literal[
    "Histone",
    "TFs and others",
    "RNA polymerase",
    "Input control",
    "ATAC-Seq",
    "DNase-seq",
    "Bisulfite-Seq",
]


class ChIPAtlasDB:
    """
    ChIP-Atlas HTTP API client.

    Зачем нужен:
    - TF binding near genes;
    - histone marks;
    - RNA polymerase / ATAC / DNase / Bisulfite data availability;
    - download URLs for peak BED files.

    Binding evidence does not by itself give activation/repression direction.
    """

    def __init__(self, timeout: int = 60) -> None:
        self.base_url = "https://chip-atlas.org"
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Low-level JSON request to ChIP-Atlas."""
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_genomes(self) -> list[str]:
        """Available genome assemblies, e.g. mm10."""
        return self.get_json("/data/list_of_genome.json")

    def list_experiment_types(self) -> list[dict[str, Any]]:
        """Static experiment classes."""
        return self.get_json("/data/list_of_experiment_types.json")

    def experiment_types(self, genome: str = "mm10", cl_class: str = "All cell types") -> Any:
        """Experiment class counts for genome/cell class."""
        return self.get_json("/data/experiment_types", {"genome": genome, "clClass": cl_class})

    def sample_types(self, genome: str = "mm10", ag_class: AgClass = "TFs and others") -> Any:
        """Cell type classes available for one experiment class."""
        return self.get_json("/data/sample_types", {"genome": genome, "agClass": ag_class})

    def antigens(
        self,
        genome: str = "mm10",
        ag_class: AgClass = "TFs and others",
        cl_class: str = "All cell types",
    ) -> Any:
        """Antigens/TFs/histone marks available for genome and cell class."""
        return self.get_json(
            "/data/chip_antigen",
            {"genome": genome, "agClass": ag_class, "clClass": cl_class},
        )

    def cell_types(
        self,
        genome: str = "mm10",
        ag_class: AgClass = "TFs and others",
        cl_class: str = "All cell types",
    ) -> Any:
        """Cell subtypes available for genome, experiment class, and cell class."""
        return self.get_json(
            "/data/cell_type",
            {"genome": genome, "agClass": ag_class, "clClass": cl_class},
        )

    def experiment_list(self) -> Any:
        """Full experiment list. Large response."""
        return self.get_json("/data/ExperimentList.json")

    def experiment(self, expid: str) -> Any:
        """Metadata for one SRX/GSM experiment."""
        return self.get_json("/data/exp_metadata.json", {"expid": expid})

    def colocalization(self, genome: str = "mm10") -> Any:
        """Precomputed colocalization index for genome."""
        return self.get_json("/data/colo_analysis.json", {"genome": genome})

    def target_genes_index(self) -> Any:
        """Index of available target-gene analyses."""
        return self.get_json("/data/target_genes_analysis.json")

    def bed_download_url(
        self,
        genome: str = "mm10",
        ag_class: AgClass = "TFs and others",
        ag_sub_class: str | None = None,
        cl_class: str = "All cell types",
        cl_sub_class: str | None = None,
        qval: str = "5",
    ) -> Any:
        """Return direct download URL for assembled peak BED file."""
        condition = {
            "genome": genome,
            "agClass": ag_class,
            "clClass": cl_class,
            "qval": qval,
        }
        if ag_sub_class:
            condition["agSubClass"] = ag_sub_class
        if cl_sub_class:
            condition["clSubClass"] = cl_sub_class

        response = requests.post(
            f"{self.base_url}/download",
            json={"condition": condition},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
