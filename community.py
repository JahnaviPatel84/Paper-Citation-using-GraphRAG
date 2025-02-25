import networkx as nx 
import os
from networkx.algorithms import community  
from cdlib.algorithms import leiden  
import matplotlib.pyplot as plt
import networkx as nx
import os
import requests
import json
import os
import re
from sklearn.cluster import DBSCAN
from clients import client
import html
from typing import Any, cast



def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    # NOTE: The import is done here to reduce the initial import time of the module
    from graspologic.utils import largest_connected_component

    graph = graph.copy()
    graph = cast("nx.Graph", largest_connected_component(graph))
    graph = normalize_node_names(graph)
    return _stabilize_graph(graph)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    # If the graph is undirected, we create the edges in a stable way, so we get the same results
    # for example:
    # A -> B
    # in graph theory is the same as
    # B -> A
    # in an undirected graph
    # however, this can lead to downstream issues because sometimes
    # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
    # but they base some of their logic on the order of the nodes, so the order ends up being important
    # so we sort the nodes in the edge in a stable way, so that we always get the same order
    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    fixed_graph.add_edges_from(edges)
    return fixed_graph


def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    return nx.relabel_nodes(graph, node_mapping)


def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
):
    """Apply a hierarchical clustering algorithm to a graph."""
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result = {}
        clusters[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = raw_community_id
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    # NOTE: This import is done here to reduce the initial import time of the graphrag package
    from graspologic.partition import hierarchical_leiden

    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy

def detect_communities(graph):
    communities = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:
            try:
                sub_communities = leiden(subgraph)
                for community in sub_communities.communities:
                    communities.append(list(community))
            except Exception as e:
                print(f"Error processing community: {e}")
        else:
            communities.append(list(subgraph.nodes))
    return communities


# 5. Graph Communities → Community Summaries
def summarize_communities(communities, graph):
    community_summaries = []
    for index, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize the following community of entities and relationships."},
                {"role": "user", "content": description}
            ]
        )
        summary = response.choices[0].message.content.strip()
        community_summaries.append(summary)
    return community_summaries

