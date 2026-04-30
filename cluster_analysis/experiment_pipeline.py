import networkx as nx
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "results/results/gen_res"
GROUND_TRUTH_FILE = "facebook.txt"

B = 100
N = 120
ALPHA = 0.05

MODELS = ["ER", "ChungLu", "SBM", "Kronecker"]
METHODS = ["EDGEIND", "LOCLBDG", "PARABDG"]

DATASET_NAME = "facebook"

# =========================================================
# LOAD GRAPH
# =========================================================

def load_edge_list(path):
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, v = line.strip().split()[:2]
            if u != v:
                G.add_edge(int(u), int(v))
    return nx.convert_node_labels_to_integers(G)

def sample_subgraph(G, n=N):
    nodes = np.random.choice(list(G.nodes()), n, replace=False)
    return G.subgraph(nodes).copy()

# =========================================================
# METRICS
# =========================================================

def triangle_stats(G):
    tri = nx.triangles(G)
    tri_total = sum(tri.values()) // 3
    wedges = sum(d*(d-1)//2 for _, d in G.degree())
    return tri_total, wedges

def gcc(G):
    tri_total, wedges = triangle_stats(G)
    return 3 * tri_total / wedges if wedges > 0 else 0

def alcc(G):
    tri = nx.triangles(G)
    vals = [tri[u] / (d*(d-1)/2) for u, d in G.degree() if d >= 2]
    return np.mean(vals) if vals else 0

def k4(G, samples=1500):
    nodes = list(G.nodes)
    n = len(nodes)
    if n < 4:
        return 0

    count = 0
    for _ in range(samples):
        quad = np.random.choice(nodes, 4, replace=False)
        if G.subgraph(quad).number_of_edges() == 6:
            count += 1

    return count * (n*(n-1)*(n-2)*(n-3)/24) / samples

def C3(G):
    tri = nx.triangles(G)
    k4_est = k4(G)
    W3 = sum(tri[u]*(d-2) for u, d in G.degree() if d >= 2)
    return (12 * k4_est) / W3 if W3 > 0 else 0

metric_functions = {
    "gcc": gcc,
    "alcc": alcc,
    "k4": k4,
    "C3": C3,
}

# =========================================================
# SAFE MONTE CARLO TEST (WITH PERCENTILES)
# =========================================================

def monte_carlo_test(x0, sims):
    sims = np.array(sims)

    if len(sims) == 0:
        return {
            "real": x0,
            "mean_sim": np.nan,
            "std_sim": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "reject": False,
            "note": "NO_SIM_DATA"
        }

    n_le = np.sum(sims <= x0)
    n_ge = np.sum(sims >= x0)

    p = min(1.0, 2 * min(
        (n_le + 1) / (len(sims) + 1),
        (n_ge + 1) / (len(sims) + 1)
    ))

    ci_low, ci_high = np.percentile(sims, [2.5, 97.5])

    return {
        "real": x0,
        "mean_sim": np.mean(sims),
        "std_sim": np.std(sims),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "reject": p <= ALPHA
    }

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_analysis(G_real):

    raw_rows = []
    test_rows = []

    for model in MODELS:
        for method in METHODS:

            print(f"\nMODEL={model} METHOD={method}")

            G0 = sample_subgraph(G_real)
            real_metrics = {m: f(G0) for m, f in metric_functions.items()}

            sim_metrics = {m: [] for m in metric_functions}

            for _ in tqdm(range(B)):

                # IMPORTANT: replace with your real generator if needed
                G = sample_subgraph(G_real)

                stats = {m: f(G) for m, f in metric_functions.items()}

                raw_rows.append({
                    "dataset": DATASET_NAME,
                    "model": model,
                    "method": method,
                    **stats
                })

                for m in stats:
                    sim_metrics[m].append(stats[m])

            for m in metric_functions:
                res = monte_carlo_test(real_metrics[m], sim_metrics[m])

                test_rows.append({
                    "dataset": DATASET_NAME,
                    "model": model,
                    "method": method,
                    "metric": m,
                    **res
                })

    return pd.DataFrame(raw_rows), pd.DataFrame(test_rows)

# =========================================================
# VISUAL 1: SWIMLANE + STAR + CI
# =========================================================

def plot_swimlane(raw_df, test_df, metric="gcc"):

    plt.figure(figsize=(11, 6))

    y_map = {}
    labels = []
    idx = 0

    for model in MODELS:
        for method in METHODS:
            key = f"{model}+{method}"
            y_map[key] = idx
            labels.append(key)
            idx += 1

    # -------------------------
    # synthetic simulation dots (transparent)
    # -------------------------
    for _, row in raw_df.iterrows():
        key = f"{row['model']}+{row['method']}"
        plt.scatter(
            row[metric],
            y_map[key],
            color="black",
            s=20
        )

    # -------------------------
    # real + CI + star marker
    # -------------------------
    for _, row in test_df[test_df["metric"] == metric].iterrows():
        key = f"{row['model']}+{row['method']}"
        y = y_map[key]

        # CI band
        plt.plot(
            [row["ci_low"], row["ci_high"]],
            [y, y],
            linewidth=3
        )

        # REAL (star)
        plt.scatter(
            row["real"],
            y,
            marker="*",
            s=180,
            color="black",
            label="Real (G₀)" if y == 0 else ""
        )

    plt.yticks(list(y_map.values()), list(y_map.keys()))
    plt.xlabel(metric.upper())
    plt.ylabel("Pipeline (Model + Method)")
    plt.title(f"{metric.upper()} - Hypothesis Test with CI + Simulation Spread")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# VISUAL 2: PERCENTILE BAND VIEW (NEW)
# =========================================================

def plot_percentile_band(test_df, metric="gcc"):

    plt.figure(figsize=(10, 5))

    subset = test_df[test_df["metric"] == metric]

    y = np.arange(len(subset))

    # percentile bars
    plt.hlines(y, subset["ci_low"], subset["ci_high"], linewidth=6)

    # real values
    plt.scatter(subset["real"], y, marker="*", s=150, color="black")

    plt.yticks(y, [f"{m}-{mo}-{me}" for m, mo, me in zip(subset["model"], subset["method"], subset["metric"])])

    plt.xlabel(metric.upper())
    plt.title(f"{metric.upper()} Percentile Bands (2.5%–97.5%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    G_real = load_edge_list(GROUND_TRUTH_FILE)

    raw_df, test_df = run_analysis(G_real)

    print("\n===== FINAL HYPOTHESIS TABLE =====")
    print(test_df[["model","method","metric","real","ci_low","ci_high","p_value","reject"]])

    raw_df.to_csv("raw_metrics.csv", index=False)
    test_df.to_csv("hypothesis_results.csv", index=False)

    plot_swimlane(raw_df, test_df, metric="gcc")
    plot_percentile_band(test_df, metric="gcc")
