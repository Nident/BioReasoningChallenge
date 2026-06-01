import networkx as nx
from io import StringIO
import requests
import pandas as pd

class GetPath:
    def __init__(self, pert, target, required_score=400, add_nodes=20):
        self.pert = pert
        self.target = target
        self.required_score = required_score
        self.add_nodes = add_nodes or []

        self.STRING_URL = "https://string-db.org/api"
        self.SPECIES = 10090  # mouse

    def string_get(self, endpoint, params):
        params = dict(params)
        params["caller_identity"] = "mlgenx_nident"
        r = requests.get(f"{self.STRING_URL}/tsv/{endpoint}", params=params)
        r.raise_for_status()
        return r.text

    def map_string_ids(self, genes, species=None):
        species = species or self.SPECIES
        text = self.string_get("get_string_ids", {
            "identifiers": "\r".join(genes),
            "species": species,
            "limit": 1,
            "echo_query": 1,
        })
        return pd.read_csv(StringIO(text), sep="\t")
    
    def build_string_graph(self, edges):
        G = nx.Graph()

        for _, row in edges.iterrows():
            a = row["preferredName_A"]
            b = row["preferredName_B"]

            score = row.get("score", None)
            if score is None and "combined_score" in row:
                score = row["combined_score"]

            G.add_edge(a, b, score=score)

        return G

    def get_network_interactions(self, string_ids, species=None, required_score=None, add_nodes=None):
        species = species or self.SPECIES
        required_score = required_score or self.required_score
        add_nodes = add_nodes or self.add_nodes

        text = self.string_get("network", {
            "identifiers": "\r".join(string_ids),
            "species": species,
            "required_score": required_score,
            "add_nodes": add_nodes,
            "network_type": "functional",
        })
        return pd.read_csv(StringIO(text), sep="\t")
    
    def get_path(self):
        from time import sleep

        mapped = self.map_string_ids([self.pert, self.target], species=self.SPECIES)

        if mapped.empty:
            print({
                "pert": self.pert,
                "target": self.target,
                "status": "mapping_failed",
                "evidence": "STRING mapping failed for one or both genes."
            })

        mapping = dict(zip(mapped["queryItem"], mapped["stringId"]))

        if self.pert not in mapping or self.target not in mapping:
            print({
                "pert": self.pert,
                "target": self.target,
                "status": "mapping_failed",
                "evidence": "STRING mapping failed for one or both genes."
            })

        edges = self.get_network_interactions(
            [mapping[self.pert], mapping[self.target]],
            species=self.SPECIES,
            required_score=self.required_score,
            add_nodes=self.add_nodes,
        )
        # print("Edges", edges)
        # sleep(1)  # be nice to the STRING API


        if edges.empty:
            print({
                "pert": self.pert,
                "target": self.target,
                "status": "no_edges",
                "evidence": "No STRING network edges returned."
            })

        G = self.build_string_graph(edges)

        result = {
            "pert": self.pert,
            "target": self.target,
            "status": "ok",
            "direct_edge": G.has_edge(self.pert, self.target),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }

        # print("G", G)
        # sleep(1)  # be nice to the STRING API

        if G.has_edge(self.pert, self.target):
            result["direct_score"] = G[self.pert][self.target].get("score")

        try:
            path = nx.shortest_path(G, source=self.pert, target=self.target)
            result["shortest_path"] = path
            result["path_length"] = len(path) - 1
        except Exception:
            result["shortest_path"] = None
            result["path_length"] = None

        return result



if __name__ == "__main__":
    pert = "Cebpb"
    target = "Brd8dc"

    gp = GetPath(pert, target)
    result = gp.get_path()
    print(result)