from __future__ import annotations

from io import StringIO
from typing import Literal

import pandas as pd
import requests


TRRUSTSpecies = Literal["mouse", "human"]


class TRRUSTDB:
    """
    TRRUST v2 client.

    Зачем нужен:
    - manually curated TF-target regulatory edges;
    - mouse and human regulatory relationships;
    - mode field can include Activation/Repression/Unknown.

    Это direct evidence для:
    TF -> activates/represses -> target gene.
    """

    def __init__(
        self,
        species: TRRUSTSpecies = "mouse",
        timeout: float | tuple[float, float] = (5.0, 10.0),
    ) -> None:
        self.species = species
        self.timeout = timeout
        self.url = f"https://www.grnpedia.org/trrust/data/trrust_rawdata.{species}.tsv"

    def raw(self) -> pd.DataFrame:
        """Download full TRRUST table for selected species."""
        response = requests.get(self.url, timeout=self.timeout)
        response.raise_for_status()
        return pd.read_csv(
            StringIO(response.text),
            sep="\t",
            names=["tf", "target", "mode", "pmid"],
        )

    def by_tf(self, tf: str) -> pd.DataFrame:
        """All targets regulated by one TF."""
        df = self.raw()
        return df[df["tf"] == tf]

    def by_target(self, target: str) -> pd.DataFrame:
        """All TFs known to regulate one target gene."""
        df = self.raw()
        return df[df["target"] == target]

    def pair(self, tf: str, target: str) -> pd.DataFrame:
        """Direct TF-target record if present."""
        df = self.raw()
        return df[(df["tf"] == tf) & (df["target"] == target)]
