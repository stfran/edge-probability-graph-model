from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import orbit_count


# GCD-11 uses the 11 non-redundant graphlet orbits for 2- to 4-node graphlets.
# Orbit indices follow ORCA's node orbit ordering.
GCD11_ORBITS: Sequence[int] = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]


def _progress(iterable, *, desc: str, show_progress: bool):
    return tqdm(
        iterable,
        desc=desc,
        disable=not show_progress,
        leave=False,
        dynamic_ncols=True,
    )


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: str | Path, *, show_progress: bool = True) -> nx.Graph:
    """
    Load an undirected, unweighted graph from one of the formats that are
    common in the EPGM repository:

    - edge lists in ../data/gt_txt/*.txt
    - edge lists exported from generated results
    - pickled / gpickle-style NetworkX graphs in ../data/nx_graph/*

    The returned graph is converted to a simple undirected graph, self-loops
    are removed, and nodes are relabeled to consecutive integers.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".edgelist", ".edges", ".csv", ""}:
        G = nx.Graph()
        with open(path, "r", encoding="utf-8") as f:
            for line in _progress(f, desc=f"Loading {path.name}", show_progress=show_progress):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                u, v = parts[0], parts[1]
                if u != v:
                    G.add_edge(u, v)
        return _normalize_graph(G)

    if suffix in {".gpickle", ".pickle", ".pkl"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError(f"Unsupported pickle payload type: {type(obj)!r}")
        return _normalize_graph(nx.Graph(obj))

    raise ValueError(f"Unsupported graph format: {path}")


def _normalize_graph(G: nx.Graph) -> nx.Graph:
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


# ---------------------------------------------------------------------------
# Triangle-based metrics
# ---------------------------------------------------------------------------

def unique_triangle_count(G: nx.Graph) -> int:
    tri_per_node = nx.triangles(G)
    return sum(tri_per_node.values()) // 3


def wedge_count(G: nx.Graph) -> int:
    return sum(math.comb(d, 2) for _, d in G.degree() if d >= 2)


def gcc(G: nx.Graph) -> float:
    """Global clustering coefficient, using the same wedge-based definition as the paper."""
    w = wedge_count(G)
    if w == 0:
        return 0.0
    return 3 * (unique_triangle_count(G) / w)


def alcc(G: nx.Graph) -> float:
    """
    Repo / paper-operational definition:
    nodes with degree < 2 contribute 0, then divide by all nodes.
    """
    tri = nx.triangles(G)
    total = 0.0
    n = G.number_of_nodes()
    for u, d in G.degree():
        if d >= 2:
            total += tri[u] / math.comb(d, 2)
    return total / n if n else 0.0


def alcc_other(G: nx.Graph) -> float:
    """
    Alternative definition: average only over nodes where local clustering is defined.
    This is not the operational definition used for the reproduced paper metrics.
    """
    tri = nx.triangles(G)
    vals = []
    for u, d in G.degree():
        if d >= 2:
            vals.append(tri[u] / math.comb(d, 2))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Clique enumeration utilities
# ---------------------------------------------------------------------------

def _ordered_neighbors(G: nx.Graph) -> Tuple[list[int], dict[int, int], dict[int, set[int]]]:
    nodes = sorted(G.nodes())
    order = {u: i for i, u in enumerate(nodes)}
    nbrs_fwd = {u: {v for v in G.neighbors(u) if order[v] > order[u]} for u in nodes}
    return nodes, order, nbrs_fwd


def enumerate_k_cliques(
    G: nx.Graph,
    k: int,
    *,
    show_progress: bool = True,
) -> Iterator[Tuple[int, ...]]:
    """
    Enumerate all k-cliques exactly once using a forward-neighbor ordering.
    Suitable for exact counts on modest graphs.
    """
    if k < 2:
        raise ValueError("k must be >= 2")

    nodes, order, nbrs_fwd = _ordered_neighbors(G)

    def extend(prefix: Tuple[int, ...], candidates: set[int], target_size: int):
        if len(prefix) == target_size:
            yield prefix
            return

        cand_list = sorted(candidates, key=lambda x: order[x])
        for i, u in enumerate(cand_list):
            remaining = len(cand_list) - i
            needed = target_size - len(prefix)
            if remaining < needed:
                break
            next_candidates = candidates.intersection(nbrs_fwd[u])
            yield from extend(prefix + (u,), next_candidates, target_size)
            candidates.remove(u)

    for u in _progress(nodes, desc=f"Processing nodes for {k}-cliques", show_progress=show_progress):
        yield from extend((u,), set(nbrs_fwd[u]), k)


def count_k_cliques(G: nx.Graph, k: int, *, show_progress: bool = True) -> int:
    return sum(1 for _ in enumerate_k_cliques(G, k, show_progress=show_progress))


def k_clique_density_and_count(
    G: nx.Graph,
    k: int,
    *,
    show_progress: bool = True,
) -> Tuple[float, int]:
    n = G.number_of_nodes()
    if n < k:
        return 0.0, 0
    denom = math.comb(n, k)
    if denom == 0:
        return 0.0, 0
    cliques = count_k_cliques(G, k, show_progress=show_progress)
    density = cliques / denom
    return density, cliques


# ---------------------------------------------------------------------------
# Higher-order clustering (Yin et al., 2018)
# ---------------------------------------------------------------------------

def node_k_clique_membership_counts(
    G: nx.Graph,
    k: int,
    *,
    show_progress: bool = True,
) -> Dict[int, int]:
    """For each node u, count how many k-cliques contain u."""
    counts: dict[int, int] = defaultdict(int)
    for clique in enumerate_k_cliques(G, k, show_progress=show_progress):
        for u in clique:
            counts[u] += 1
    for u in G.nodes():
        counts[u] += 0
    return dict(counts)


def higher_order_local_clustering(
    G: nx.Graph,
    ell: int,
    *,
    show_progress: bool = True,
) -> Dict[int, float]:
    """
    Compute local C_ell(u) exactly using Eq. (8) in Yin et al. (2018):

        C_ell(u) = |K_{ell+1}(u)| / ((d_u - ell + 1) * |K_ell(u)|)

    for nodes where the denominator is nonzero.
    """
    if ell < 2:
        raise ValueError("ell must be >= 2")

    K_ell_u = node_k_clique_membership_counts(G, ell, show_progress=show_progress)
    K_ell1_u = node_k_clique_membership_counts(G, ell + 1, show_progress=show_progress)

    out: dict[int, float] = {}
    for u, d in _progress(
        G.degree(),
        desc="Computing higher-order local clustering",
        show_progress=show_progress,
    ):
        denom = (d - ell + 1) * K_ell_u.get(u, 0)
        if denom > 0:
            out[u] = K_ell1_u.get(u, 0) / denom
    return out


def higher_order_average_local_clustering(
    G: nx.Graph,
    ell: int,
    *,
    show_progress: bool = True,
) -> float:
    vals = list(higher_order_local_clustering(G, ell, show_progress=show_progress).values())
    return float(np.mean(vals)) if vals else 0.0


def higher_order_global_clustering(
    G: nx.Graph,
    ell: int,
    *,
    show_progress: bool = True,
) -> float:
    """
    Compute global C_ell exactly:

        C_ell = ((ell^2 + ell) * |K_{ell+1}|) / |W_ell|

    where

        |W_ell| = sum_u |K_ell(u)| * (d_u - ell + 1)

    following Eqs. (4), (7), and (9) in Yin et al. (2018).
    """
    if ell < 2:
        raise ValueError("ell must be >= 2")

    num_cliques = count_k_cliques(G, ell + 1, show_progress=show_progress)
    K_ell_u = node_k_clique_membership_counts(G, ell, show_progress=show_progress)

    W_ell = 0
    for u, d in _progress(
        G.degree(),
        desc="Computing higher-order global clustering",
        show_progress=show_progress,
    ):
        if d >= ell - 1:
            W_ell += K_ell_u.get(u, 0) * (d - ell + 1)

    if W_ell == 0:
        return 0.0

    return ((ell * ell + ell) * num_cliques) / W_ell


# ---------------------------------------------------------------------------
# GCD-11 via orbit-count
# ---------------------------------------------------------------------------

def node_orbits_4(G: nx.Graph) -> np.ndarray:
    """
    Compute node orbit counts up to 4-node graphlets using orbit-count.

    Returns
    -------
    np.ndarray
        Shape (n_nodes, 15), matching the standard ORCA node-orbit output
        ordering for graphlets up to size 4.
    """
    node_list = list(sorted(G.nodes()))
    arr = orbit_count.node_orbit_counts(
        G,
        graphlet_size=4,
        node_list=node_list,
    )
    arr = np.asarray(arr, dtype=np.int64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != 15:
        raise ValueError(f"Expected 15 orbit columns for graphlet_size=4, got shape {arr.shape}")

    return arr


def gcm11_from_orbits(orbits_15: np.ndarray) -> np.ndarray:
    """Construct the 11x11 Graphlet Correlation Matrix used in GCD-11."""
    X = pd.DataFrame(orbits_15[:, GCD11_ORBITS])
    gcm = X.corr(method="spearman").fillna(0.0).to_numpy(dtype=float)
    return gcm


def gcd11(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Compute the Graphlet Correlation Distance using the 11 non-redundant
    node orbits for 2- to 4-node graphlets.
    """
    O1 = node_orbits_4(G1)
    O2 = node_orbits_4(G2)

    GCM1 = gcm11_from_orbits(O1)
    GCM2 = gcm11_from_orbits(O2)

    iu = np.triu_indices_from(GCM1, k=1)
    v1 = GCM1[iu]
    v2 = GCM2[iu]

    return float(np.linalg.norm(v1 - v2))


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_graph(
    G: nx.Graph,
    *,
    compute_k5: bool = False,
    compute_c4: bool = False,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Compute the default bundle of graph metrics for one graph."""
    k4_density, k4_count = k_clique_density_and_count(G, 4, show_progress=show_progress)

    out: Dict[str, float] = {
        "n_nodes": float(G.number_of_nodes()),
        "n_edges": float(G.number_of_edges()),
        "n_triangles": float(unique_triangle_count(G)),
        "n_wedges": float(wedge_count(G)),
        "gcc": gcc(G),
        "alcc": alcc(G),
        "k4_count": float(k4_count),
        "k4_density": float(k4_density),
        "C3_global": higher_order_global_clustering(G, 3, show_progress=show_progress),
        "C3_avg_local": higher_order_average_local_clustering(G, 3, show_progress=show_progress),
    }

    if compute_k5:
        k5_density, k5_count = k_clique_density_and_count(G, 5, show_progress=show_progress)
        out["k5_count"] = float(k5_count)
        out["k5_density"] = float(k5_density)

    if compute_c4:
        out["C4_global"] = higher_order_global_clustering(G, 4, show_progress=show_progress)
        out["C4_avg_local"] = higher_order_average_local_clustering(G, 4, show_progress=show_progress)

    return out
