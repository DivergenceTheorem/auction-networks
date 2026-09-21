import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cdlib import NodeClustering, evaluation


def internal_external_degree(G, module):
    internal = 0
    external = 0
    for node in module:
        for neighbor in G.neighbors(node):
            if neighbor in module:
                internal += 1
            else:
                external += 1
    return internal, external


def compute_fitness(G, module, alpha):
    K_in, K_out = internal_external_degree(G, module)
    if K_in + K_out == 0:
        return 0
    return K_in / ((K_in + K_out) ** alpha)


def compute_node_fitness(G, module, node, alpha):
    return compute_fitness(G, module | {node}, alpha) - compute_fitness(
        G, module, alpha
    )  # Fitness(module_with_node) - Fitness(module_without_node)


def detect_natural_community(G, node, alpha=1.0):
    module = {node}  # Start with community which is equal to our node
    neighbors = set(G.neighbors(node))  # Find the neighbors of the node

    while True:
        if not neighbors:  # Stop if we don't see any neighbors
            break

        fitness_gains = {
            neighbor: compute_node_fitness(G, module, neighbor, alpha)
            # Computing node fitness of each neighbor of G. Format -> node: fitness_value
            for neighbor in neighbors
        }

        best_neighbor = max(
            fitness_gains, key=fitness_gains.get
        )  # Pick the neighbor with largest fitness and add it to the community
        if fitness_gains[best_neighbor] <= 0:
            break  # If it so happens that even the maximum value for fitness of neighbors is non-poisitve, then we stop the algorithm

        module.add(best_neighbor)  # Add our best neighbor to module
        neighbors.remove(
            best_neighbor
        )  # Remove it from the list of neighbors of G, because it's know a part of G
        neighbors.update(
            set(G.neighbors(best_neighbor)) - module
        )  # Add neighbors of best neighbor, because these neighbors are now neighbors of G

        while True:
            node_fitnesses = {n: compute_node_fitness(G, module, n, alpha) for n in module}
            for n, f in node_fitnesses.items():
                if f < 0:
                    module.remove(n)
                    neighbors.add(n)
                    break  # Restart checking from scratch after a removal
            else:
                break  # No negative fitness node found -> exit inner loop

    return module


def find_multiple_communities(G, alpha=1.0, num_trials=10):
    communities = []  # Initialize the list of communities
    for _ in range(num_trials):
        seed = random.choice(list(G.nodes()))  # Pick a random node to then build a community around it
        community = detect_natural_community(G, seed, alpha)  # Find the community of that node
        if community and community not in communities:
            # Check if community is not empty and not in the list of existing communities
            communities.append(community)  # If this condition is satisfied, we add our current community to the 'communities' list
    return communities


def simplified_community_detection(G, alpha=1.0):
    """The computationally simplified algorithm"""

    communities = []  # List to store the detected communities
    assigned_nodes = set()  # Tracks the nodes already assigned to a community
    all_nodes = list(G.nodes())  # List of all nodes for sampling
    unvisited_nodes = set(all_nodes)  # Create a list of yet unvisited nodes

    while unvisited_nodes:  # While the set of unvisited nodes is not empty, we keep on iterating...
        seed = random.choice(list(unvisited_nodes))  # Pick a random unassigned node
        community = detect_natural_community(G, seed, alpha)

        if community:  # Avoid empty communities
            communities.append(community)
            assigned_nodes.update(community)
            unvisited_nodes -= community  # Remove assigned nodes from future consideration

    return communities


def get_clusters(G=None, alpha=1):
    """We get the clusters from the G using our community detection algorithm"""

    if G is None:
        G = nx.karate_club_graph()  # Set G = karate_club and alpha = 1 as deafult
    communities = simplified_community_detection(G, alpha)
    pred_clusters = [set(community) for community in communities]  # Predicted clusters (communities)
    return pred_clusters


def calculate_onmi(pred_clusters, true_clusters):

    pred_clusters = [list(c) for c in pred_clusters]
    true_clusters = [list(c) for c in true_clusters]
    gt_clustering = NodeClustering(communities=true_clusters, graph=None, method_name="ground_truth")
    pred_clustering = NodeClustering(communities=pred_clusters, graph=None, method_name="predicted")

    onmi_score = evaluation.overlapping_normalized_mutual_information_LFK(
        pred_clustering, gt_clustering
    ).score
    return onmi_score


def draw_communities(G, communities, node_to_community):
    colors = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in communities
    ]  # List of N random colors, where N is the number of communities

    node_colors = [
        colors[node_to_community.get(node, -1)] if node in node_to_community else "#cccccc"
        for node in G.nodes()
    ]

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=node_colors, with_labels=True, edge_color="gray")
    plt.title("Detected Natural Communities")
    plt.show()


def print_communities(communities):
    print("Detected communities:\n")
    for i, community in enumerate(communities):
        print(f"Community {i+1} ({len(community)} nodes): {sorted(community)}")


def alpha_sweep_onmi(G, true_clusters, alphas, seed=42):

    random.seed(seed)
    scores = {}
    for alpha in alphas:
        pred_clusters = get_clusters(G, alpha)
        scores[alpha] = calculate_onmi(pred_clusters, true_clusters)
    return scores


def best_alpha(onmi_scores):
    """Alpha(s) achieving the maximum ONMI"""

    max_onmi = max(onmi_scores.values())
    max_alphas = [a for a, v in onmi_scores.items() if v == max_onmi]
    return max_alphas, max_onmi


def plot_onmi_vs_alpha(onmi_scores, dataset_name="graph"):
    x = list(onmi_scores.keys())
    y = list(onmi_scores.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker=".")
    ax.set_xlabel(r"resolution parameter $\alpha$")
    ax.set_ylabel("ONMI vs. ground truth")
    ax.set_title(f"ONMI({dataset_name}) as a function of $\\alpha$")
    ax.grid(True)
    return fig


def average_cover_fitness(G, communities, alpha=1.0):
    if not communities:
        return 0.0
    return float(np.mean([compute_fitness(G, c, alpha) for c in communities]))


def fitness_histogram(G, alphas, seed=42, n_bins=40):
    
    random.seed(seed)
    fbar_values = []
    covers_by_fbar = {}
    for alpha in alphas:
        communities = simplified_community_detection(G, alpha)
        fbar = average_cover_fitness(G, communities, alpha=1.0)
        fbar_values.append(fbar)
        covers_by_fbar.setdefault(round(fbar, 6), (alpha, communities))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fbar_values, bins=n_bins)
    ax.set_xlabel(r"average cover fitness $\bar{f}_\mathcal{P}$")
    ax.set_ylabel("frequency")
    ax.set_title("Fitness histogram (stable covers = tall peaks)")
    return fig, fbar_values, covers_by_fbar


def overlap_fraction(G, alphas, seed=42):

    random.seed(seed)
    fractions = []
    for alpha in alphas:
        communities = simplified_community_detection(G, alpha)
        membership_count = defaultdict(int)
        for c in communities:
            for n in c:
                membership_count[n] += 1
        n_overlapping = sum(1 for n, c in membership_count.items() if c > 1)
        fractions.append(n_overlapping / G.number_of_nodes())
    return fractions


def plot_overlap_fraction(alphas, fractions, dataset_name="graph"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(alphas), fractions, marker=".")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("fraction of overlapping nodes")
    ax.set_title(f"Overlap vs. $\\alpha$ ({dataset_name})")
    ax.grid(True)
    return fig


def visualize_cover(G, true_clusters, pred_clusters, dataset_name="graph", true_group_names=None):
    """Ground truth split vs. the detected cover"""
    pos = nx.spring_layout(G, seed=42)
    nodes = list(G.nodes())

    # --- ground truth coloring ---
    node_to_true = {}
    for i, cluster in enumerate(true_clusters):
        for n in cluster:
            node_to_true[n] = i
    true_group_labels = sorted(set(node_to_true.values()))
    true_colors = [plt.cm.tab10(node_to_true.get(n, 0)) for n in nodes]

    # --- detected cover coloring (overlap-aware) ---
    membership = defaultdict(list)
    for i, c in enumerate(pred_clusters):
        for n in c:
            membership[n].append(i)

    pred_colors = []
    for n in nodes:
        comms = membership.get(n, [])
        if len(comms) == 0:
            pred_colors.append((0.8, 0.8, 0.8, 1.0))  # unassigned -> gray
        elif len(comms) == 1:
            pred_colors.append(plt.cm.tab20(comms[0] % 20))
        else:
            pred_colors.append((0.0, 0.0, 0.0, 1.0))  # overlapping -> black

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plt.sca(axes[0])
    nx.draw(G, pos, node_color=true_colors, with_labels=False, node_size=100)
    axes[0].set_title(f"Ground truth: {dataset_name}")
    true_legend = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            label=(true_group_names or {}).get(g, f"Community {g + 1}"),
            markerfacecolor=plt.cm.tab10(g), markersize=10,
        )
        for g in true_group_labels
    ]
    axes[0].legend(handles=true_legend, loc="lower right", fontsize=8)

    plt.sca(axes[1])
    nx.draw(G, pos, node_color=pred_colors, with_labels=False, node_size=100)
    axes[1].set_title(f"Lancichinetti et al. cover: {dataset_name}")

    pred_legend = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=f"Community {i + 1}",
            markerfacecolor=plt.cm.tab20(i % 20), markersize=10,
        )
        for i in range(len(pred_clusters))
    ]
    if any(len(membership.get(n, [])) == 0 for n in nodes):
        pred_legend.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label="Unassigned",
                markerfacecolor=(0.8, 0.8, 0.8, 1.0), markersize=10,
            )
        )
    if any(len(membership.get(n, [])) > 1 for n in nodes):
        pred_legend.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label="Overlapping",
                markerfacecolor=(0.0, 0.0, 0.0, 1.0), markersize=10,
            )
        )
    axes[1].legend(handles=pred_legend, loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from datasets import load_karate_club

    G, gt, true_clusters = load_karate_club()
    alphas = np.linspace(0, 5, 501)
    scores = alpha_sweep_onmi(G, true_clusters, alphas)
    alphas_best, onmi_best = best_alpha(scores)
    print("best alpha(s):", alphas_best, "  ONMI:", onmi_best)