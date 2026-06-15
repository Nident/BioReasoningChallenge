from pprint import pprint

import networkx as nx
from io import StringIO
import requests
import pandas as pd
import random
from typing import Any


STRING_EVIDENCE_COLUMNS = {
    "nscore": "neighborhood",
    "fscore": "fusion",
    "pscore": "cooccurrence",
    "ascore": "coexpression",
    "escore": "experimental",
    "dscore": "database",
    "tscore": "textmining",
}


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
    
    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if pd.isnull(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def build_string_graph(self, edges):
        G = nx.Graph()

        for _, row in edges.iterrows():
            a = row["preferredName_A"]
            b = row["preferredName_B"]

            score = row.get("score", None)
            if score is None and "combined_score" in row:
                score = row["combined_score"]

            edge_data = {
                "score": self._safe_float(score),
            }

            for column, name in STRING_EVIDENCE_COLUMNS.items():
                if column in row:
                    edge_data[column] = self._safe_float(row.get(column))
                    edge_data[name] = self._safe_float(row.get(column))

            G.add_edge(a, b, **edge_data)

        return G

    def edge_report(self, G, source, target):
        edge = G[source][target]

        return {
            "source": source,
            "target": target,
            "combined_score": edge.get("score", 0.0),
            "evidence_channels": {
                name: edge.get(name, 0.0)
                for name in STRING_EVIDENCE_COLUMNS.values()
            },
        }

    def path_report(self, G, path):
        edge_scores = []
        edge_reports = []

        for source, target in zip(path[:-1], path[1:]):
            score = G[source][target].get("score", 0.0)
            edge_scores.append(score)
            edge_reports.append(self.edge_report(G, source, target))

        confidence_product = 1.0
        for score in edge_scores:
            confidence_product *= score

        return {
            "path": path,
            "path_length": len(path) - 1,
            "edge_scores": edge_scores,
            "confidence_product": confidence_product,
            "min_edge_score": min(edge_scores) if edge_scores else 0.0,
            "sum_edge_score": sum(edge_scores),
            "edges": edge_reports,
        }
    
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

    def get_weighted_paths(self, G, source, target, max_paths=500):
        try:
            paths_gen = nx.all_simple_paths(
                G,
                source=source,
                target=target,
                cutoff=self.random_path_cutoff
            )

            reports = []
            for path in paths_gen:
                reports.append(self.path_report(G, path))

                if len(reports) >= max_paths:
                    break

            reports.sort(
                key=lambda item: (
                    item["confidence_product"],
                    item["min_edge_score"],
                    item["sum_edge_score"],
                ),
                reverse=True,
            )

            return reports[:self.top_k_paths]

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

    def get_common_neighbor_report(self, G, source, target, max_items=10):
        try:
            common_neighbors = sorted(
                set(G.neighbors(source)) & set(G.neighbors(target))
            )
        except nx.NetworkXError:
            return []

        report = []

        for neighbor in common_neighbors:
            source_score = G[source][neighbor].get("score", 0.0)
            target_score = G[neighbor][target].get("score", 0.0)

            report.append({
                "neighbor": neighbor,
                "source_edge_score": source_score,
                "target_edge_score": target_score,
                "score_product": source_score * target_score,
                "source_edge": self.edge_report(G, source, neighbor),
                "target_edge": self.edge_report(G, neighbor, target),
            })

        report.sort(
            key=lambda item: (
                item["score_product"],
                item["source_edge_score"] + item["target_edge_score"],
            ),
            reverse=True,
        )

        return report[:max_items]

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
            result["direct_edge_evidence"] = self.edge_report(
                G,
                self.pert,
                self.target,
            )

        result["pert_degree"] = G.degree(self.pert) if self.pert in G else 0
        result["target_degree"] = G.degree(self.target) if self.target in G else 0
        result["common_neighbors"] = self.get_common_neighbor_report(
            G,
            self.pert,
            self.target,
        )
        result["num_common_neighbors"] = len(result["common_neighbors"])

        try:
            top_paths = self.get_top_k_paths(G, self.pert, self.target)
            weighted_paths = self.get_weighted_paths(G, self.pert, self.target)

            result["top_paths"] = top_paths
            result["weighted_paths"] = weighted_paths

            if top_paths:
                result["shortest_path"] = top_paths[0]["path"]
                result["path_length"] = top_paths[0]["path_length"]
            else:
                result["shortest_path"] = None
                result["path_length"] = None

            if weighted_paths:
                result["best_weighted_path"] = weighted_paths[0]["path"]
                result["best_weighted_path_confidence_product"] = weighted_paths[0]["confidence_product"]
                result["best_weighted_path_min_edge_score"] = weighted_paths[0]["min_edge_score"]
            else:
                result["best_weighted_path"] = None
                result["best_weighted_path_confidence_product"] = 0.0
                result["best_weighted_path_min_edge_score"] = 0.0

            result["random_paths"] = self.get_random_paths(
                G,
                self.pert,
                self.target
            )

        except Exception:
            result["top_paths"] = []
            result["weighted_paths"] = []
            result["random_paths"] = []
            result["shortest_path"] = None
            result["path_length"] = None
            result["best_weighted_path"] = None
            result["best_weighted_path_confidence_product"] = 0.0
            result["best_weighted_path_min_edge_score"] = 0.0

        return result



if __name__ == "__main__":
    pert = "Cebpb"
    target = "Brd8dc"

    gp = GeneGraph(pert, target)
    result = gp.get_path()
    pprint(result)
