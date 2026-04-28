import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

B = 100
N = 120
ALPHA = 0.05
DATASET_NAME = "facebook"

# =========================================================
# LOAD GRAPH
# =========================================================

def load_edge_list(path):
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            u, v = line.strip().split()
            if u != v:
                G.add_edge(int(u), int(v))
    return nx.convert_node_labels_to_integers(G)

def sample_subgraph(G, n=N):
    nodes = np.random.choice(list(G.nodes()), n, replace=False)
    return G.subgraph(nodes).copy()

# =========================================================
# METRICS (SWAPPABLE LAYER)
# =========================================================

def triangle_stats(G):
    tri = nx.triangles(G)
    tri_total = sum(tri.values()) // 3
    wedges = sum(d*(d-1)//2 for _, d in G.degree())
    return tri, tri_total, wedges

def gcc(G):
    tri, tri_total, wedges = triangle_stats(G)
    return 3 * tri_total / wedges if wedges > 0 else 0

def alcc(G):
    tri = nx.triangles(G)
    vals = [
        tri[u] / (d*(d-1)/2)
        for u, d in G.degree()
        if d >= 2
    ]
    return np.mean(vals) if vals else 0

def k4(G, samples=2000):
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
# PROBABILITY MODELS
# =========================================================

def er_prob(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = (2*m)/(n*(n-1))
    return lambda u, v: p

def chung_lu_prob(G):
    deg = dict(G.degree())
    m = G.number_of_edges()
    return lambda u, v: min(1.0, (deg[u]*deg[v])/(2*m + 1e-9))

def sbm_prob(G, k=4):
    nodes = list(G.nodes())
    blocks = {u: i % k for i, u in enumerate(nodes)}

    counts = np.zeros((k, k))
    totals = np.zeros((k, k))

    for u in nodes:
        for v in nodes:
            if u >= v:
                continue
            i, j = blocks[u], blocks[v]
            totals[i][j] += 1
            totals[j][i] += 1

    for u, v in G.edges():
        i, j = blocks[u], blocks[v]
        counts[i][j] += 1
        counts[j][i] += 1

    probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return lambda u, v: probs[blocks[u]][blocks[v]]

def kronecker_prob(G):
    P = np.array([[0.9, 0.5],[0.5, 0.1]])
    n = G.number_of_nodes()
    k = int(np.log2(n))

    def prob(u, v):
        p = 1.0
        uu, vv = u, v
        for _ in range(k):
            p *= P[uu % 2, vv % 2]
            uu //= 2
            vv //= 2
        return p

    return prob

# =========================================================
# BINDING SCHEMES
# =========================================================

def sample_edgeind(n, prob):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u+1, n):
            if np.random.rand() < prob(u, v):
                G.add_edge(u, v)
    return G

def sample_loclbdg(n, prob):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    theta = np.random.rand(n)
    for u in range(n):
        for v in range(u+1, n):
            if prob(u, v) >= max(theta[u], theta[v]):
                G.add_edge(u, v)
    return G

def sample_parabdg(n, prob):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    theta = np.random.rand()
    for u in range(n):
        for v in range(u+1, n):
            if prob(u, v) >= theta:
                G.add_edge(u, v)
    return G

# =========================================================
# GENERATOR WRAPPER
# =========================================================

def generate_graph(G_real, model, method):
    if model == "ER":
        prob = er_prob(G_real)
    elif model == "ChungLu":
        prob = chung_lu_prob(G_real)
    elif model == "SBM":
        prob = sbm_prob(G_real)
    elif model == "Kronecker":
        prob = kronecker_prob(G_real)

    if method == "EDGEIND":
        return sample_edgeind(N, prob)
    elif method == "LOCLBDG":
        return sample_loclbdg(N, prob)
    elif method == "PARABDG":
        return sample_parabdg(N, prob)

# =========================================================
# MONTE CARLO TEST
# =========================================================

def monte_carlo_test(x0, sims):
    sims = np.array(sims)

    n_le = np.sum(sims <= x0)
    n_ge = np.sum(sims >= x0)

    p = min(1.0, 2 * min((n_le+1)/(len(sims)+1),
                         (n_ge+1)/(len(sims)+1)))

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

    models = ["ER", "ChungLu", "SBM", "Kronecker"]
    methods = ["EDGEIND", "LOCLBDG", "PARABDG"]

    raw_rows = []
    test_rows = []

    for model in models:
        for method in methods:

            print(f"\nMODEL={model} METHOD={method}")

            G0 = sample_subgraph(G_real)

            real_metrics = {m: f(G0) for m, f in metric_functions.items()}
            sim_metrics = {m: [] for m in metric_functions}

            for _ in tqdm(range(B)):

                G = generate_graph(G_real, model, method)

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
# PLOTTING (WITH LEGEND)
# =========================================================

def plot_results(raw_df, test_df):

    plt.figure(figsize=(8,6))

    colors = {
        "ER": "red",
        "ChungLu": "blue",
        "SBM": "green",
        "Kronecker": "purple"
    }

    for model in raw_df["model"].unique():
        subset = raw_df[raw_df["model"] == model]
        plt.scatter(
            subset["gcc"],
            subset["C3"],
            label=model,
            alpha=0.6,
            color=colors.get(model, "black")
        )

    plt.xlabel("GCC")
    plt.ylabel("C3")
    plt.title("Graph Metric Distribution (Synthetic vs Real)")
    plt.legend()
    plt.show()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    G_real = load_edge_list("facebook.txt")

    raw_df, test_df = run_analysis(G_real)

    print("\n===== HYPOTHESIS RESULTS =====")
    print(test_df)

    raw_df.to_csv("raw_metrics.csv", index=False)
    test_df.to_csv("hypothesis_results.csv", index=False)

    plot_results(raw_df, test_df)
