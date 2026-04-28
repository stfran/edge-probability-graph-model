import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import tempfile
import os
import time
import matplotlib.pyplot as plt

# =========================================================
# CONFIG: ORCA PATH
# =========================================================

ORCA_PATH = "wsl ./orca/orca"

# =========================================================
# LOAD REAL DATASET
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

    G = nx.convert_node_labels_to_integers(G)

    print(f"\nLoaded dataset: {path}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    return G

# =========================================================
# FAST CORE METRICS
# =========================================================

def triangle_stats(G):
    tri = nx.triangles(G)
    tri_total = sum(tri.values()) // 3
    wedges = sum(d*(d-1)//2 for _, d in G.degree())
    return tri, tri_total, wedges

def gcc_from_stats(tri_total, wedges):
    return 3 * tri_total / wedges if wedges > 0 else 0

def alcc_fast(G, tri):
    vals = []
    for u, d in G.degree():
        if d >= 2:
            vals.append(tri[u] / (d*(d-1)/2))
    return np.mean(vals) if vals else 0

# =========================================================
# K4 (HYBRID)
# =========================================================

def k4_exact(G):
    count = 0
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 4:
            count += 1
        elif len(clique) > 4:
            break
    return count

def k4_sample(G, samples=5000):
    nodes = list(G.nodes)
    n = len(nodes)
    if n < 4:
        return 0

    count = 0
    for _ in range(samples):
        quad = np.random.choice(nodes, 4, replace=False)
        if G.subgraph(quad).number_of_edges() == 6:
            count += 1

    return count * (n*(n-1)*(n-2)*(n-3) / 24) / samples

def k4_hybrid(G, threshold=120):
    return k4_exact(G) if G.number_of_nodes() <= threshold else k4_sample(G)

# =========================================================
# HIGHER-ORDER CLUSTERING
# =========================================================

def C3_global_fast(G, tri_total, k4):
    tri = nx.triangles(G)

    W3 = 0
    for u, d in G.degree():
        if d >= 2:
            W3 += tri[u] * (d - 2)

    if W3 == 0:
        return 0

    return (12 * k4) / W3

# =========================================================
# ORCA
# =========================================================

def node_orbits_orca(G):
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        in_path = f.name
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

    out_path = in_path + ".out"

    try:
        subprocess.run(
            [ORCA_PATH, "node", "4", in_path, out_path],
            check=True,
            capture_output=True
        )
        orbits = np.loadtxt(out_path)
    except Exception as e:
        print("⚠️ ORCA failed:", e)
        orbits = None

    os.remove(in_path)
    if os.path.exists(out_path):
        os.remove(out_path)

    return orbits

# =========================================================
# GCD-11
# =========================================================

GCD11_ORBITS = [0,1,2,4,5,6,7,8,9,10,11]

def gcd11(G1, G2):
    O1 = node_orbits_orca(G1)
    O2 = node_orbits_orca(G2)

    if O1 is None or O2 is None:
        return np.nan

    X1 = pd.DataFrame(O1[:, GCD11_ORBITS])
    X2 = pd.DataFrame(O2[:, GCD11_ORBITS])

    GCM1 = X1.corr(method="spearman").fillna(0).to_numpy()
    GCM2 = X2.corr(method="spearman").fillna(0).to_numpy()

    iu = np.triu_indices_from(GCM1, k=1)
    return np.linalg.norm(GCM1[iu] - GCM2[iu])

# =========================================================
# GRAPH MODELS
# =========================================================

def generate_er(n):
    return nx.erdos_renyi_graph(n, 0.05)

def generate_chung_lu(n):
    degrees = np.random.zipf(2.5, n)
    degrees = np.clip(degrees, 1, n//2)
    w = degrees / np.sum(degrees)
    return nx.expected_degree_graph(w * n, selfloops=False)

def generate_sbm(n, k=4):
    sizes = [n // k] * k
    probs = np.full((k, k), 0.01)
    np.fill_diagonal(probs, 0.2)
    return nx.stochastic_block_model(sizes, probs)

def generate_kronecker(n):
    P = np.array([[0.9, 0.5],
                  [0.5, 0.1]])

    k = int(np.log2(n))
    G = nx.Graph()

    for i in range(n):
        for j in range(i+1, n):
            prob = 1.0
            ii, jj = i, j

            for _ in range(k):
                prob *= P[ii % 2, jj % 2]
                ii //= 2
                jj //= 2

            if np.random.rand() < prob:
                G.add_edge(i, j)

    G.add_nodes_from(range(n))
    return G

# =========================================================
# SAMPLERS
# =========================================================

def identity(G):
    return G

def edge_dropout(G, p=0.5):
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for u, v in G.edges():
        if np.random.rand() < p:
            H.add_edge(u, v)
    return H

def degree_bias_sampling(G):
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    n = len(G)

    for u, v in G.edges():
        deg_factor = (G.degree(u) + G.degree(v)) / (2*n)
        p = min(1.0, 2 * deg_factor)

        if np.random.rand() < p:
            H.add_edge(u, v)

    return H

# =========================================================
# SUMMARY
# =========================================================

def summarize_graph_fast(G):
    tri, tri_total, wedges = triangle_stats(G)

    gcc = gcc_from_stats(tri_total, wedges)
    alcc = alcc_fast(G, tri)
    k4 = k4_hybrid(G)
    C3 = C3_global_fast(G, tri_total, k4)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "triangles": tri_total,
        "gcc": gcc,
        "alcc": alcc,
        "k4": k4,
        "C3": C3
    }

# =========================================================
# EXPERIMENT LOOP
# =========================================================

def run_experiment_fast(model_fn, sampler_fn, n=120, B=10):
    results = []
    start = time.time()

    for i in tqdm(range(B), desc="Trials"):
        t0 = time.time()

        G = model_fn(n)
        H = sampler_fn(G)

        stats = summarize_graph_fast(H)
        results.append(stats)

        print(f"[Trial {i}] gcc={stats['gcc']:.4f} "
              f"C3={stats['C3']:.6f} k4={stats['k4']:.1f} "
              f"time={time.time()-t0:.2f}s")

    print(f"Finished in {time.time()-start:.2f}s\n")
    return pd.DataFrame(results)

# =========================================================
# FULL EXPERIMENT
# =========================================================

def full_experiment(models, samplers):
    all_results = []
    total_start = time.time()

    for m_name, m_fn in models.items():
        for s_name, s_fn in samplers.items():

            print("="*50)
            print(f"MODEL: {m_name} | SAMPLER: {s_name}")
            print("="*50)

            df = run_experiment_fast(m_fn, s_fn)

            df["model"] = m_name
            df["method"] = s_name

            all_results.append(df)

    print(f"\nTOTAL TIME: {time.time()-total_start:.2f}s")
    return pd.concat(all_results)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------
    # REAL GRAPH
    # -----------------------------
    real_graph = load_edge_list("../data/gt_txt/facebook.txt")

    # -----------------------------
    # MODELS
    # -----------------------------
    models = {
        "ER": generate_er,
        "ChungLu": generate_chung_lu,
        "SBM": generate_sbm,
        "Kronecker": generate_kronecker,
    }

    samplers = {
        "base": identity,
        "dropout": edge_dropout,
        "degree_bias": degree_bias_sampling,
    }

    # -----------------------------
    # SYNTHETIC EXPERIMENTS
    # -----------------------------
    df_models = full_experiment(models, samplers)

    # -----------------------------
    # REAL GRAPH EXPERIMENTS
    # -----------------------------
    print("\n" + "="*50)
    print("REAL DATASET EXPERIMENT")
    print("="*50)

    real_results = []

    for s_name, s_fn in samplers.items():
        H = s_fn(real_graph)
        stats = summarize_graph_fast(H)

        stats["model"] = "Facebook"
        stats["method"] = s_name

        print(f"Sampler={s_name} | GCC={stats['gcc']:.4f} | C3={stats['C3']:.6f}")

        real_results.append(stats)

    df_real = pd.DataFrame(real_results)

    # -----------------------------
    # COMBINE
    # -----------------------------
    df = pd.concat([df_models, df_real], ignore_index=True)

    print("\n===== SUMMARY =====")
    print(df.groupby(["model", "method"]).agg(["mean", "std"]))

    df.to_csv("results_with_real.csv", index=False)
    print("\nSaved results_with_real.csv")

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure()

    colors = df["model"].astype("category").cat.codes

    plt.scatter(df["gcc"], df["C3"], c=colors)
    plt.xlabel("GCC")
    plt.ylabel("C3")
    plt.title("Synthetic vs Real Graph Structure")

    plt.show()

    # -----------------------------
    # GCD COMPARISON
    # -----------------------------
    print("\n===== GCD vs REAL GRAPH =====")

    for name, model_fn in models.items():
        G_model = model_fn(real_graph.number_of_nodes())
        dist = gcd11(G_model, real_graph)
        print(f"{name} vs Facebook GCD: {dist:.4f}")
