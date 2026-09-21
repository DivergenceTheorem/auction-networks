import csv
import os
from collections import defaultdict

import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _ground_truth_to_clusters(ground_truth):
    community_dict = defaultdict(list)
    for node, label in ground_truth.items():
        community_dict[label].append(node)
    return [set(community) for community in community_dict.values()]


def load_karate_club():
    "Load ZKC dataset and return configuration, ground truth and true clusters"

    G = nx.karate_club_graph()
    ground_truth = {node: G.nodes[node]["club"] for node in G.nodes}
    true_clusters = _ground_truth_to_clusters(ground_truth)
    return G, ground_truth, true_clusters


def load_dolphins():
    "Load Dolphins dataset and return configuration, ground truth and true clusters"

    gml_path = os.path.join(DATA_DIR, "dolphins.gml")
    G = nx.read_gml(gml_path)
    G = nx.convert_node_labels_to_integers(G, label_attribute="dolphin_label")
    ground_truth = {node: G.nodes[node]["gt"] for node in G.nodes}
    true_clusters = _ground_truth_to_clusters(ground_truth)
    return G, ground_truth, true_clusters


def load_polbooks():
    "Load Political Books dataset and return configuration, ground truth and true clusters"

    gml_path = os.path.join(DATA_DIR, "polbooks.gml")
    G = nx.read_gml(gml_path)
    G = nx.convert_node_labels_to_integers(G, label_attribute="book_title")
    ground_truth = {node: G.nodes[node]["value"] for node in G.nodes}
    true_clusters = _ground_truth_to_clusters(ground_truth)
    return G, ground_truth, true_clusters


DATASETS = {
    "karate": {
        "loader": load_karate_club,
        "display_name": "Zachary's Karate Club",
        "label_names": {"Mr. Hi": "Mr. Hi", "Officer": "Officer"},
        "k": 2,
    },
    "dolphins": {
        "loader": load_dolphins,
        "display_name": "Dolphin Social Network",
        "label_names": {"Male": "Male", "Female": "Female", "Unknown": "Unknown"},
        "k": 2,
    },
    "polbooks": {
        "loader": load_polbooks,
        "display_name": "Books about US Politics",
        "label_names": {"l": "Liberal", "n": "Neutral", "c": "Conservative"},
        "k": 3,
    },
}


if __name__ == "__main__":
    for key, spec in DATASETS.items():
        G, gt, true_clusters = spec["loader"]()
        print(
            f"{spec['display_name']:28s} nodes={G.number_of_nodes():4d} "
            f"edges={G.number_of_edges():4d} groups={[len(c) for c in true_clusters]}"
        )