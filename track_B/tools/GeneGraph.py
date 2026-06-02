import networkx as nx
from io import StringIO
import requests
import pandas as pd
import random
from typing import Any


class GeneGraph:
    def __init__(self, pert, target, required_score=400, add_nodes=20,
                 n_random_paths=3, random_path_cutoff=5, top_k_paths=3):
        self.pert = pert
        self.target = target
        self.required_score = required_score
        self.add_nodes = add_nodes
        self.n_random_paths = n_random_paths
        self.top_k_paths = top_k_paths
        self.random_path_cutoff = random_path_cutoff

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
    
    def get_top_k_paths(self, G, source, target):
        try:
            paths_gen = nx.shortest_simple_paths(G, source=source, target=target)
            paths = []

            for path in paths_gen:
                paths.append({
                    "path": path,
                    "path_length": len(path) - 1
                })

                if len(paths) >= self.top_k_paths:
                    break

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        

    def get_random_paths(self, G, source, target):
        try:
            all_paths_gen = nx.all_simple_paths(
                G,
                source=source,
                target=target,
                cutoff=self.random_path_cutoff
            )

            paths = list(all_paths_gen)

            if not paths:
                return []

            sampled = random.sample(paths, min(self.n_random_paths, len(paths)))

            return [
                {
                    "path": path,
                    "path_length": len(path) - 1
                }
                for path in sampled
            ]

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_network_interactions(
        self,
        string_ids,
        species=None,
        required_score=None,
        add_nodes=None,
    ):
        species = species or self.SPECIES
        required_score = required_score or self.required_score
        add_nodes = self.add_nodes if add_nodes is None else add_nodes

        text = self.string_get("network", {
            "identifiers": "\r".join(string_ids),
            "species": species,
            "required_score": required_score,
            "add_nodes": add_nodes,
            "network_type": "functional",
        })
        return pd.read_csv(StringIO(text), sep="\t")
    
    def get_path(self) -> dict[str, Any]:
        mapped = self.map_string_ids([self.pert, self.target], species=self.SPECIES)

        if mapped.empty:
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "mapping_failed",
                "evidence": "STRING mapping failed for one or both genes."
            }

        mapping = dict(zip(mapped["queryItem"], mapped["stringId"]))

        if self.pert not in mapping or self.target not in mapping:
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "mapping_failed",
                "evidence": "STRING mapping failed for one or both genes."
            }

        edges = self.get_network_interactions(
            [mapping[self.pert], mapping[self.target]],
            species=self.SPECIES,
            required_score=self.required_score,
            add_nodes=self.add_nodes,
        )
        # print("Edges", edges)
        # sleep(1)  # be nice to the STRING API


        if edges.empty:
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "no_edges",
                "evidence": "No STRING network edges returned."
            }

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
            top_paths = self.get_top_k_paths(G, self.pert, self.target)

            result["top_paths"] = top_paths

            if top_paths:
                result["shortest_path"] = top_paths[0]["path"]
                result["path_length"] = top_paths[0]["path_length"]
            else:
                result["shortest_path"] = None
                result["path_length"] = None

            result["random_paths"] = self.get_random_paths(
                G,
                self.pert,
                self.target
            )

        except Exception:
            result["top_paths"] = []
            result["random_paths"] = []
            result["shortest_path"] = None
            result["path_length"] = None

        return result



if __name__ == "__main__":
    pert = "Cebpb"
    target = "Brd8dc"

    gp = GeneGraph(pert, target)
    result = gp.get_path()
    print(result)
