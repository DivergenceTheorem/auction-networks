# Social Network Community Detection

Code, data and term paper for a CSCI 447 "Machine Learning: Theory and Practice" term project comparing two community-detection approaches on three network-science benchmarks. This repository is the community-detection component of a larger, ongoing project, **Collusion and Cartel Detection in Auction Markets: Theory, Networks, and Algorithms**, which extends beyond the CSCI 447 course scope with additional theoretical and algorithmic work; only the course-scoped benchmarking study is contained here.

Two algorithms are implemented and compared:
- **K-means (Lloyd's algorithm)** on a 3-D node embedding (degree, clustering coefficient, betweenness centrality).
- **Lancichinetti et al. (2009)**: an overlapping community-detection method that grows each node's "natural community" by local optimization of a fitness function `f_G = k_in / (k_in + k_out)^α`.

Each method is scored against known ground-truth communities using Overlapping Normalized Mutual Information (ONMI). For Lancichinetti's method, the resolution parameter α is swept over a range and the best-case (highest-ONMI) partition is reported, i.e. α is selected using the same ground truth the method is scored against, not picked blind.

| Dataset | Nodes | Edges | Ground truth | K-means (NMI) | Lancichinetti, best-case (ONMI) |
|---|---|---|---|---|---|
| Zachary's Karate Club | 34 | 78 | 2 factions ('Mr. Hi' / 'Officer') | 0.0062 | 0.6905 |
| Dolphin Social Network | 62 | 159 | 2-way structural split | 0.0153 | 0.8768 |
| Political Books | 105 | 441 | 3 classes (liberal / neutral / conservative) | 0.0189 | 0.4338 |

---

## Repository layout

```
.
├── requirements.txt          # pinned Python dependencies
├── datasets.py                # load_karate_club / load_dolphins / load_polbooks
├── kmeans_clustering.py        # K-means baseline (feature extraction + clustering + plots)
├── lancichinetti.py            # fitness function, natural-community search, cover construction, ONMI
├── run_experiment.py           # runs both methods on all 3 datasets, writes output/
├── term_paper.tex              # paper (compiles to term_paper.pdf)
├── reference.bib               # bibliography
├── data/
│   ├── dolphins.gml
│   └── polbooks.gml
└── output/                     # figures + table1.txt, written by run_experiment.py
```

---

## 1. Environment setup

Developed on **Python 3.13**.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| networkx | graphs, GML loading, layout |
| numpy | resolution-parameter sweeps |
| matplotlib | all figures |
| scikit-learn | K-means, ARI/NMI |
| cdlib | ONMI (`overlapping_normalized_mutual_information_LFK`) |

---

## 2. Reproduce the results

```bash
python run_experiment.py
```

This runs both methods on all three datasets and writes every figure plus `table1.txt` to `output/`.

To rebuild the paper:

```bash
latexmk -pdf term_paper.tex
```

---

## 3. What the code does

### K-means baseline (`kmeans_clustering.py`)

Each node is embedded as a 3-D feature vector (degree, local clustering coefficient, betweenness centrality), then clustered with `sklearn.cluster.KMeans(random_state=42)`. Evaluated with Adjusted Rand Index and Normalized Mutual Information against ground truth.

### Lancichinetti et al. (2009) (`lancichinetti.py`)

For a given node, its natural community is grown greedily: at each step, the neighboring node with the highest fitness contribution is added, any member with negative fitness is dropped, and this repeats until no neighbor improves the fitness. `simplified_community_detection` repeats this from random unassigned seed nodes until every node has a community, producing a cover (nodes may belong to more than one community — overlap).

`alpha_sweep_onmi` runs this for a range of α and scores each resulting cover against ground truth with ONMI; `best_alpha` returns the α(s) achieving the maximum. `visualize_cover` plots ground truth vs. detected cover side by side, with detected nodes colored by community, gray for unassigned, and black for overlapping.

### Datasets (`datasets.py`)

- `load_karate_club()` — `networkx.karate_club_graph()`; ground truth is the built-in `'club'` attribute.
- `load_dolphins()` — `data/dolphins.gml`; ground truth is the `'gt'` attribute, a 2-way structural split (**not** gender — the original notebook this project built on assumed gender labels, but the surviving dataset only encodes a generic binary split).
- `load_polbooks()` — `data/polbooks.gml` (Newman's political-books network); ground truth is the `'value'` attribute (`l`/`n`/`c`).

---

## 4. Output data dictionary (`output/`)

| File(s) | Content |
|---|---|
| `fig2/3/4_*_kmeans.png` | Ground truth vs. K-means partition, per dataset |
| `fig5*_onmi_vs_alpha.png` | ONMI as a function of α |
| `fig5*_fitness_histogram.png` | Distribution of average cover fitness across the α sweep |
| `fig6*_cover.png` | Ground truth vs. Lancichinetti cover at the best-case α |
| `fig7*_overlap.png` | Fraction of overlapping nodes vs. α |
| `table1.txt` | K-means NMI and Lancichinetti best-case ONMI per dataset |

---

## 5. Reproducibility notes

- All random draws are seeded: `random.seed(42)` at the start of each `alpha_sweep_onmi` call, `KMeans(random_state=42)`, `spring_layout(seed=42)`.
- `load_dolphins` relabels nodes to integers on load. Without this, Python's per-process string-hash randomization makes `set(nodes)` iteration order (and therefore which node `simplified_community_detection` starts from) non-deterministic across runs, despite the fixed seed. `load_karate_club` and `load_polbooks` are unaffected since their nodes are already integers.
- The reported Lancichinetti ONMI is the best-case score over the α sweep (α chosen using the same ground truth it's scored against).

---

## Citation

> Tursynkhan, A., Kim, A., Kurmanov, M., Yerzhanova, Z., & Skakov, M. (2025). *Social Network Community Detection.*

### References

- Lancichinetti, A., Fortunato, S., & Kertész, J. (2009). Detecting the overlapping and hierarchical community structure in complex networks. *New Journal of Physics*, 11(3), 033015.
- Newman, M. (n.d.). Network data. <http://www-personal.umich.edu/~mejn/netdata/>
