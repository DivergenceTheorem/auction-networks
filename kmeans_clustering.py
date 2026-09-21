import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def extract_features(G):
    """
    Accepts network configuration and outputs coordianates of network nodes in the 3D feature space defined by
    degree, local clustering coefficient, betweenness centrality of the node.
    """

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G)

    nodes = list(G.nodes())
    features = [[degree[n], clustering[n], betweenness[n]] for n in nodes]
    return nodes, np.array(features)


def run_kmeans(
    G,
    ground_truth,
    k,
    label_to_int,
    dataset_name="graph",
    true_group_names=None,
    random_state=42,
    eval_nodes=None,
):
    """Run the 3d-feature K-means community detection on graph G.

    Parameters
    ----------
    G : networkx.Graph
    ground_truth : dict {node: label}
    k : number of clusters
    label_to_int : maps each ground-truth label string to an integer
    eval_nodes : nodes that have defined ground truth, if 'None' then we assume all nodes have defined ground truth.

    Returns
    -------
    dict with predicted labels, ARI, NMI, and the node ordering used.
    """
    nodes, X = extract_features(G) # compute 3d feature for each node

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    predicted_labels = kmeans.fit_predict(X)

    G = G.copy()
    for i, node in enumerate(nodes):
        G.nodes[node]["cluster"] = int(predicted_labels[i]) # assign kmeans clusters for each node in the network G

    # --- compute across only those nodes that have defined ground truths - eval_nodes. All nodes have gt by default.
    if eval_nodes is None:
        eval_idx = range(len(nodes))
    else:
        eval_nodes = set(eval_nodes)
        eval_idx = [i for i, n in enumerate(nodes) if n in eval_nodes]

    true_labels_int = [label_to_int[ground_truth[nodes[i]]] for i in eval_idx]
    predicted_eval = [predicted_labels[i] for i in eval_idx]

    ari = adjusted_rand_score(true_labels_int, predicted_eval)
    nmi = normalized_mutual_info_score(true_labels_int, predicted_eval)

    print(f"[{dataset_name}] K-means (k={k})  ARI = {ari:.4f}   NMI = {nmi:.4f}")

    _plot_true_vs_kmeans(
        G, nodes, ground_truth, predicted_labels, dataset_name, true_group_names
    )

    return {
        "nodes": nodes,
        "predicted_labels": predicted_labels,
        "ari": ari,
        "nmi": nmi,
        "graph_with_clusters": G,
    }


def _plot_true_vs_kmeans(G, nodes, ground_truth, predicted_labels, dataset_name, true_group_names):
    """
    Side-by-side comparison of ground truth vs. K-means partition. 
    Reproduces the Fig. 2 / Fig. 3 / Fig. 4 layout of the paper.
    """

    pos = nx.spring_layout(G, seed=42)
    
    true_group_list = sorted(set(ground_truth.values()))
    true_color_map = {g: plt.cm.tab10(i) for i, g in enumerate(true_group_list)}
    true_colors = [true_color_map[ground_truth[n]] for n in nodes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plt.sca(axes[0])
    nx.draw(G, pos, node_color=true_colors, with_labels=False, node_size=100)
    axes[0].set_title(f"True groups: {dataset_name}")
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            label=(true_group_names or {}).get(g, str(g)),
            markerfacecolor=true_color_map[g], markersize=10,
        )
        for g in true_group_list
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.sca(axes[1])
    nx.draw(G, pos, node_color=predicted_labels, cmap=plt.cm.Set1, with_labels=False, node_size=100)
    axes[1].set_title(f"K-Means Clustering Result: {dataset_name}")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from datasets import load_karate_club, load_dolphins, load_polbooks

    G, gt, _ = load_karate_club()
    run_kmeans(
        G, gt, k=2, label_to_int={"Mr. Hi": 0, "Officer": 1},
        dataset_name="Karate Club",
        true_group_names={"Mr. Hi": "Mr. Hi", "Officer": "Officer"},
    )

    G, gt, _ = load_dolphins()
    known_nodes = [n for n, g in gt.items() if g in ("1", "2")]
    run_kmeans(
        G, gt, k=2, label_to_int={"1": 0, "2": 1},
        dataset_name="Dolphins",
        true_group_names={"1": "Group 1", "2": "Group 2"},
        eval_nodes=known_nodes,
    )

    G, gt, _ = load_polbooks()
    run_kmeans(
        G, gt, k=3, label_to_int={"l": 0, "n": 1, "c": 2},
        dataset_name="Political Books",
        true_group_names={"l": "Liberal", "n": "Neutral", "c": "Conservative"},
    )
