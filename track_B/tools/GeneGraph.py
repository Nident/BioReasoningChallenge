from pprint import pprint

import logging
import networkx as nx
from io import StringIO
import requests
import pandas as pd
import random
import time
from typing import Any


logger = logging.getLogger(__name__)


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
    def __init__(self, pert, target, required_score=400, add_nodes=200,
                 n_random_paths=3, random_path_cutoff=5, top_k_paths=3,
                 request_timeout=30, max_path_candidates=500):
        self.pert = pert
        self.target = target
        self.required_score = required_score
        self.add_nodes = add_nodes
        self.n_random_paths = n_random_paths
        self.top_k_paths = top_k_paths
        self.random_path_cutoff = random_path_cutoff
        self.request_timeout = request_timeout
        self.max_path_candidates = max_path_candidates

        self.STRING_URL = "https://string-db.org/api"
        self.SPECIES = 10090  # mouse

        logger.info(
            "Initialized GeneGraph pert=%s target=%s required_score=%s add_nodes=%s "
            "n_random_paths=%s random_path_cutoff=%s top_k_paths=%s "
            "request_timeout=%s max_path_candidates=%s",
            pert,
            target,
            required_score,
            add_nodes,
            n_random_paths,
            random_path_cutoff,
            top_k_paths,
            request_timeout,
            max_path_candidates,
        )

    def string_get(self, endpoint, params):
        params = dict(params)
        params["caller_identity"] = "mlgenx_nident"

        start = time.monotonic()
        logger.info(
            "STRING request start endpoint=%s identifiers=%s required_score=%s add_nodes=%s",
            endpoint,
            str(params.get("identifiers")).replace("\r", ","),
            params.get("required_score"),
            params.get("add_nodes"),
        )

        try:
            r = requests.get(
                f"{self.STRING_URL}/tsv/{endpoint}",
                params=params,
                timeout=self.request_timeout,
            )
        except requests.RequestException:
            logger.exception(
                "STRING request failed endpoint=%s elapsed=%.2fs",
                endpoint,
                time.monotonic() - start,
            )
            raise

        r.raise_for_status()
        logger.info(
            "STRING request done endpoint=%s status=%s bytes=%s elapsed=%.2fs",
            endpoint,
            r.status_code,
            len(r.text),
            time.monotonic() - start,
        )
        return r.text

    def map_string_ids(self, genes, species=None):
        species = species or self.SPECIES
        start = time.monotonic()
        logger.info("Mapping genes to STRING IDs genes=%s species=%s", genes, species)
        text = self.string_get("get_string_ids", {
            "identifiers": "\r".join(genes),
            "species": species,
            "limit": 1,
            "echo_query": 1,
        })
        mapped = pd.read_csv(StringIO(text), sep="\t")
        logger.info(
            "Mapping done genes=%s rows=%s elapsed=%.2fs",
            genes,
            len(mapped),
            time.monotonic() - start,
        )
        return mapped
    
    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if pd.isnull(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def build_string_graph(self, edges):
        start = time.monotonic()
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

        logger.info(
            "Built STRING graph nodes=%s edges=%s source_rows=%s elapsed=%.2fs",
            G.number_of_nodes(),
            G.number_of_edges(),
            len(edges),
            time.monotonic() - start,
        )
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
        start = time.monotonic()
        logger.info(
            "Searching top shortest paths source=%s target=%s top_k=%s graph_nodes=%s graph_edges=%s",
            source,
            target,
            self.top_k_paths,
            G.number_of_nodes(),
            G.number_of_edges(),
        )
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

            logger.info(
                "Top shortest path search done source=%s target=%s paths=%s elapsed=%.2fs",
                source,
                target,
                len(paths),
                time.monotonic() - start,
            )
            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.info(
                "Top shortest path search found no path source=%s target=%s elapsed=%.2fs",
                source,
                target,
                time.monotonic() - start,
            )
            return []

    def get_weighted_paths(self, G, source, target, max_paths=None):
        max_paths = self.max_path_candidates if max_paths is None else max_paths
        start = time.monotonic()
        logger.info(
            "Searching weighted simple paths source=%s target=%s cutoff=%s max_candidates=%s",
            source,
            target,
            self.random_path_cutoff,
            max_paths,
        )
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
                    logger.warning(
                        "Weighted path search hit candidate limit source=%s target=%s limit=%s",
                        source,
                        target,
                        max_paths,
                    )
                    break

            reports.sort(
                key=lambda item: (
                    item["confidence_product"],
                    item["min_edge_score"],
                    item["sum_edge_score"],
                ),
                reverse=True,
            )

            result = reports[:self.top_k_paths]
            logger.info(
                "Weighted path search done source=%s target=%s candidates=%s returned=%s elapsed=%.2fs",
                source,
                target,
                len(reports),
                len(result),
                time.monotonic() - start,
            )
            return result

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.info(
                "Weighted path search found no path source=%s target=%s elapsed=%.2fs",
                source,
                target,
                time.monotonic() - start,
            )
            return []

    def get_random_paths(self, G, source, target):
        start = time.monotonic()
        logger.info(
            "Sampling random simple paths source=%s target=%s cutoff=%s max_candidates=%s sample_size=%s",
            source,
            target,
            self.random_path_cutoff,
            self.max_path_candidates,
            self.n_random_paths,
        )
        try:
            all_paths_gen = nx.all_simple_paths(
                G,
                source=source,
                target=target,
                cutoff=self.random_path_cutoff
            )

            sampled = []
            candidates_seen = 0
            truncated = False

            for path in all_paths_gen:
                candidates_seen += 1

                if len(sampled) < self.n_random_paths:
                    sampled.append(path)
                else:
                    replacement_index = random.randrange(candidates_seen)
                    if replacement_index < self.n_random_paths:
                        sampled[replacement_index] = path

                if candidates_seen >= self.max_path_candidates:
                    truncated = True
                    logger.warning(
                        "Random path sampling hit candidate limit source=%s target=%s limit=%s",
                        source,
                        target,
                        self.max_path_candidates,
                    )
                    break

            if not sampled:
                logger.info(
                    "Random path sampling found no path source=%s target=%s elapsed=%.2fs",
                    source,
                    target,
                    time.monotonic() - start,
                )
                return []

            result = [
                {
                    "path": path,
                    "path_length": len(path) - 1
                }
                for path in sampled
            ]
            logger.info(
                "Random path sampling done source=%s target=%s candidates_seen=%s returned=%s truncated=%s elapsed=%.2fs",
                source,
                target,
                candidates_seen,
                len(result),
                truncated,
                time.monotonic() - start,
            )
            return result

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.info(
                "Random path sampling found no path source=%s target=%s elapsed=%.2fs",
                source,
                target,
                time.monotonic() - start,
            )
            return []

    def get_common_neighbor_report(self, G, source, target, max_items=10):
        start = time.monotonic()
        try:
            common_neighbors = sorted(
                set(G.neighbors(source)) & set(G.neighbors(target))
            )
        except nx.NetworkXError:
            logger.info(
                "Common neighbor search failed source=%s target=%s elapsed=%.2fs",
                source,
                target,
                time.monotonic() - start,
            )
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

        result = report[:max_items]
        logger.info(
            "Common neighbor search done source=%s target=%s total=%s returned=%s elapsed=%.2fs",
            source,
            target,
            len(common_neighbors),
            len(result),
            time.monotonic() - start,
        )
        return result

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

        start = time.monotonic()
        logger.info(
            "Fetching STRING network string_ids=%s species=%s required_score=%s add_nodes=%s",
            string_ids,
            species,
            required_score,
            add_nodes,
        )
        text = self.string_get("network", {
            "identifiers": "\r".join(string_ids),
            "species": species,
            "required_score": required_score,
            "add_nodes": add_nodes,
            "network_type": "functional",
        })
        edges = pd.read_csv(StringIO(text), sep="\t")
        logger.info(
            "Fetched STRING network rows=%s elapsed=%.2fs",
            len(edges),
            time.monotonic() - start,
        )
        return edges
    
    def get_path(self) -> dict[str, Any]:
        start = time.monotonic()
        logger.info(
            "GeneGraph start pert=%s target=%s species=%s required_score=%s add_nodes=%s cutoff=%s top_k=%s",
            self.pert,
            self.target,
            self.SPECIES,
            self.required_score,
            self.add_nodes,
            self.random_path_cutoff,
            self.top_k_paths,
        )

        try:
            mapped = self.map_string_ids([self.pert, self.target], species=self.SPECIES)
        except requests.RequestException as exc:
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "string_request_failed",
                "evidence": f"STRING mapping request failed: {exc}"
            }

        if mapped.empty:
            logger.warning("STRING mapping returned no rows pert=%s target=%s", self.pert, self.target)
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "mapping_failed",
                "evidence": "STRING mapping failed for one or both genes."
            }

        mapping = dict(zip(mapped["queryItem"], mapped["stringId"]))
        logger.info("STRING mapping result pert=%s target=%s mapping=%s", self.pert, self.target, mapping)

        if self.pert not in mapping or self.target not in mapping:
            logger.warning(
                "STRING mapping missing requested gene pert=%s target=%s mapping_keys=%s",
                self.pert,
                self.target,
                sorted(mapping),
            )
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "mapping_failed",
                "evidence": "STRING mapping failed for one or both genes."
            }

        try:
            edges = self.get_network_interactions(
                [mapping[self.pert], mapping[self.target]],
                species=self.SPECIES,
                required_score=self.required_score,
                add_nodes=self.add_nodes,
            )
        except requests.RequestException as exc:
            return {
                "pert": self.pert,
                "target": self.target,
                "status": "string_request_failed",
                "evidence": f"STRING network request failed: {exc}"
            }
        # print("Edges", edges)
        # sleep(1)  # be nice to the STRING API


        if edges.empty:
            logger.info(
                "GeneGraph no edges pert=%s target=%s elapsed=%.2fs",
                self.pert,
                self.target,
                time.monotonic() - start,
            )
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
                result["best_weighted_path_report"] = weighted_paths[0]
            else:
                result["best_weighted_path"] = None
                result["best_weighted_path_confidence_product"] = 0.0
                result["best_weighted_path_min_edge_score"] = 0.0
                result["best_weighted_path_report"] = None

            result["random_paths"] = self.get_random_paths(
                G,
                self.pert,
                self.target
            )

        except Exception:
            logger.exception("Path search failed pert=%s target=%s", self.pert, self.target)
            result["top_paths"] = []
            result["weighted_paths"] = []
            result["random_paths"] = []
            result["shortest_path"] = None
            result["path_length"] = None
            result["best_weighted_path"] = None
            result["best_weighted_path_confidence_product"] = 0.0
            result["best_weighted_path_min_edge_score"] = 0.0
            result["best_weighted_path_report"] = None

        logger.info(
            "GeneGraph done pert=%s target=%s status=%s direct_edge=%s nodes=%s edges=%s elapsed=%.2fs",
            self.pert,
            self.target,
            result.get("status"),
            result.get("direct_edge"),
            result.get("num_nodes"),
            result.get("num_edges"),
            time.monotonic() - start,
        )
        return result



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pert = "Rngtt"
    target = "Saa3"

    gp = GeneGraph(pert, target)
    result = gp.get_path()
    pprint(result)

    with open("graph.json", "w") as f:
        import json
        json.dump([result], f, indent=2)
