import os

import matplotlib

matplotlib.use("Agg")  # headless: just save PNGs, don't try to open a window
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dolphins, load_karate_club, load_polbooks
from kmeans_clustering import run_kmeans
from lancichinetti import (
    alpha_sweep_onmi,
    best_alpha,
    fitness_histogram,
    get_clusters,
    overlap_fraction,
    plot_onmi_vs_alpha,
    plot_overlap_fraction,
    visualize_cover,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(fig, name):
    '''helper func to save the figure'''

    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    results = {}  # dataset -> {"kmeans_nmi":..., "lanci_onmi":..., "best_alpha":...}

    # ------------------------------------------------------------------
    # 1) Zachary's Karate Club
    # ------------------------------------------------------------------
    print("\n=== Karate Club ===")
    G, gt, true_clusters = load_karate_club()

    km = run_kmeans(
        G, gt, k=2, label_to_int={"Mr. Hi": 0, "Officer": 1},
        dataset_name="Karate Club",
        true_group_names={"Mr. Hi": "Mr. Hi", "Officer": "Officer"},
    )
    savefig(plt.gcf(), "fig2_karate_kmeans.png")

    alphas = np.linspace(0, 2, 201)
    scores = alpha_sweep_onmi(G, true_clusters, alphas)
    savefig(plot_onmi_vs_alpha(scores, "Karate Club"), "fig5a_karate_onmi_vs_alpha.png")
    alphas_best, onmi_best = best_alpha(scores)
    print(f"  best alpha(s) = {alphas_best}   ONMI = {onmi_best:.3f}")

    fig, fbar_values, covers = fitness_histogram(G, alphas)
    savefig(fig, "fig5b_karate_fitness_histogram.png")

    pred_clusters = get_clusters(G, alphas_best[0])
    savefig(
        visualize_cover(G, true_clusters, pred_clusters, "Karate Club",
                         {"Mr. Hi": "Mr. Hi", "Officer": "Officer"}),
        "fig6a_karate_cover.png",
    )

    overlap_alphas = np.linspace(0.6, 1.5, 91)
    fracs = overlap_fraction(G, overlap_alphas)
    savefig(plot_overlap_fraction(overlap_alphas, fracs, "Karate Club"), "fig7a_karate_overlap.png")

    results["Karate Club"] = {
        "kmeans_nmi": km["nmi"], "lanci_onmi": onmi_best, "best_alpha": alphas_best,
    }

    # ------------------------------------------------------------------
    # 2) Dolphin Social Network
    # ------------------------------------------------------------------
    print("\n=== Dolphins ===")
    G, gt, _ = load_dolphins()
    known_nodes = [n for n, g in gt.items() if g in ("1", "2")]
    true_clusters_known = [
        {n for n in known_nodes if gt[n] == "1"},
        {n for n in known_nodes if gt[n] == "2"},
    ]

    km = run_kmeans(
        G, gt, k=2, label_to_int={"1": 0, "2": 1},
        dataset_name="Dolphins",
        true_group_names={"1": "Group 1", "2": "Group 2"},
        eval_nodes=known_nodes,
    )
    savefig(plt.gcf(), "fig3_dolphins_kmeans.png")

    alphas = np.linspace(0.5, 1.3, 161)
    scores = alpha_sweep_onmi(G, true_clusters_known, alphas)
    savefig(plot_onmi_vs_alpha(scores, "Dolphins"), "fig5c_dolphins_onmi_vs_alpha.png")
    alphas_best, onmi_best = best_alpha(scores)
    print(f"  best alpha(s) = {alphas_best}   ONMI = {onmi_best:.3f}")

    fig, fbar_values, covers = fitness_histogram(G, alphas)
    savefig(fig, "fig5d_dolphins_fitness_histogram.png")

    pred_clusters = get_clusters(G, alphas_best[0])
    savefig(
        visualize_cover(G, true_clusters_known, pred_clusters, "Dolphins",
                         {"1": "Group 1", "2": "Group 2"}),
        "fig6b_dolphins_cover.png",
    )

    overlap_alphas = np.linspace(0.6, 1.5, 91)
    fracs = overlap_fraction(G, overlap_alphas)
    savefig(plot_overlap_fraction(overlap_alphas, fracs, "Dolphins"), "fig7b_dolphins_overlap.png")

    results["Dolphins"] = {
        "kmeans_nmi": km["nmi"], "lanci_onmi": onmi_best, "best_alpha": alphas_best,
    }

    # ------------------------------------------------------------------
    # 3) Books about US Politics
    # ------------------------------------------------------------------
    print("\n=== Political Books ===")
    G, gt, true_clusters = load_polbooks()

    km = run_kmeans(
        G, gt, k=3, label_to_int={"l": 0, "n": 1, "c": 2},
        dataset_name="Political Books",
        true_group_names={"l": "Liberal", "n": "Neutral", "c": "Conservative"},
    )
    savefig(plt.gcf(), "fig4_polbooks_kmeans.png")

    alphas = np.linspace(0.5, 1.3, 81)  # coarser: 105-node graph, keep runtime reasonable
    scores = alpha_sweep_onmi(G, true_clusters, alphas)
    savefig(plot_onmi_vs_alpha(scores, "Political Books"), "fig5e_polbooks_onmi_vs_alpha.png")
    alphas_best, onmi_best = best_alpha(scores)
    print(f"  best alpha(s) = {alphas_best}   ONMI = {onmi_best:.3f}")

    fig, fbar_values, covers = fitness_histogram(G, alphas)
    savefig(fig, "fig5f_polbooks_fitness_histogram.png")

    pred_clusters = get_clusters(G, alphas_best[0])
    savefig(
        visualize_cover(G, true_clusters, pred_clusters, "Political Books",
                         {"l": "Liberal", "n": "Neutral", "c": "Conservative"}),
        "fig6c_polbooks_cover.png",
    )

    overlap_alphas = np.linspace(0.6, 1.5, 61)
    fracs = overlap_fraction(G, overlap_alphas)
    savefig(plot_overlap_fraction(overlap_alphas, fracs, "Political Books"), "fig7c_polbooks_overlap.png")

    results["Political Books"] = {
        "kmeans_nmi": km["nmi"], "lanci_onmi": onmi_best, "best_alpha": alphas_best,
    }

    # ------------------------------------------------------------------
    # Table I  (K-means NMI  vs.  Lancichinetti et al. ONMI)
    # ------------------------------------------------------------------
    print("\n=== Table I (reconstructed) ===")
    header = f"{'Dataset':22s} {'K-means (NMI)':>15s} {'Lancichinetti (ONMI)':>22s}"
    print(header)
    with open(os.path.join(OUT_DIR, "table1.txt"), "w") as f:
        f.write(header + "\n")
        for name, r in results.items():
            line = f"{name:22s} {r['kmeans_nmi']:15.4f} {r['lanci_onmi']:22.4f}"
            print(line)
            f.write(line + "\n")

    print(f"\nAll figures + table1.txt written to {OUT_DIR}/")


if __name__ == "__main__":
    main()