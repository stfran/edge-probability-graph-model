from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import metrics as M

# =========================================================
# CONFIG
# =========================================================

# Use cached raw_metrics.csv / hypothesis_results.csv when available.
RESUME_FROM_CSV = True
OVERWRITE_RESULTS = False

# Save after this many generated graphs. Set to 1 for best interruption safety.
CHECKPOINT_EVERY_N_GRAPHS = 50

# None runs all datasets.
# To run subsets, use any of:
# ["facebook", "bio-CE-PG", "bio-SC-HT", "hamsterster", "polblogs", "web-spam"]
DATASETS = ["facebook"]
#DATASETS = None

BASE_DIR = Path(__file__).resolve().parent
GT_DIR = BASE_DIR.parent / "data" / "gt_txt"
EDGE_IND_RESULTS_DIR = BASE_DIR / "results" / "data"
BIND_RESULTS_DIR = BASE_DIR / "results" / "gen_res"
ANALYSIS_DIR = BASE_DIR / "analysis"

RAW_OUT = ANALYSIS_DIR / "raw_metrics.csv"
TEST_OUT = ANALYSIS_DIR / "hypothesis_results.csv"

RAW_KEY_COLS = ["dataset", "model_key", "method", "trial_key"]
TEST_KEY_COLS = ["dataset", "model_key", "method", "metric"]

EXPECTED_B = 100  # warn if a generated-results directory has a different count
ALPHA = 0.05

MODELS = [
    "ER",
    "CL",
    "SBM",
    "KR",
]

MODELS_DISPLAY = {
    "ER": "Erdős–Rényi",
    "CL": "Chung–Lu",
    "SBM": "Stochastic Block Model",
    "KR": "Kronecker",
}

METHODS = [
    "EDGEIND",
    "LOCLBDG",
    "PARABDG",
]

# Smoke test example: METRIC_NAMES = ["gcc"]
METRIC_NAMES = [
    "gcc",
    "alcc",
    "k4_count",
    "k4_density",
    "C3_global",
    "C3_avg_local",
]

TQDM_KW = {
    "dynamic_ncols": True,
    "mininterval": 0.5,
}

# =========================================================
# GENERATED GRAPH DIRECTORY LOOKUP
# =========================================================

RESULT_DIRS = {
    # Edge-independent baseline results
    ("CL", "EDGEIND"): [EDGE_IND_RESULTS_DIR / "orig_cl"],
    ("ER", "EDGEIND"): [EDGE_IND_RESULTS_DIR / "orig_er"],
    ("KR", "EDGEIND"): [EDGE_IND_RESULTS_DIR / "orig_kr"],
    ("SBM", "EDGEIND"): [EDGE_IND_RESULTS_DIR / "orig_sbm"],

    # Local binding
    ("CL", "LOCLBDG"): [BIND_RESULTS_DIR / "CL_iter" / "t1"],
    ("ER", "LOCLBDG"): [BIND_RESULTS_DIR / "ER_iter" / "triangle"],
    ("KR", "LOCLBDG"): [BIND_RESULTS_DIR / "KR_iter" / "t1"],
    ("SBM", "LOCLBDG"): [BIND_RESULTS_DIR / "SBM_iter" / "t1"],

    # Parallel binding
    ("CL", "PARABDG"): [BIND_RESULTS_DIR / "CL_iid" / "t1"],
    ("ER", "PARABDG"): [BIND_RESULTS_DIR / "ER_iid" / "triangle"],
    ("KR", "PARABDG"): [BIND_RESULTS_DIR / "KR_iid" / "t1"],
    ("SBM", "PARABDG"): [BIND_RESULTS_DIR / "SBM_iid" / "t1"],
}


def generated_root(model: str, method: str) -> Path:
    try:
        candidates = RESULT_DIRS[(model, method)]
    except KeyError as exc:
        raise KeyError(f"No result-directory mapping for model={model}, method={method}") from exc

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No generated-result directory found for model={model}, method={method}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def trial_id_from_path(path: str | Path) -> int | None:
    """Extract trial id from names like facebook_0.txt or bio-SC-HT_96.txt."""
    match = re.search(r"_(\d+)$", Path(path).stem)
    return int(match.group(1)) if match else None


def trial_key_from_path(path: str | Path) -> str:
    """
    Stable key for a generated graph.

    Prefer the numeric trial id from names like facebook_17.txt.
    Fall back to the filename if no numeric suffix exists.
    """
    trial = trial_id_from_path(path)
    return str(trial) if trial is not None else Path(path).name


def natural_trial_sort_key(path: str | Path):
    trial = trial_id_from_path(path)
    return (trial is None, trial if trial is not None else Path(path).name)


def generated_graph_paths(dataset: str, model: str, method: str) -> list[Path]:
    """
    Find generated graph files for one dataset/model/method combination.

    Expected naming pattern:
        facebook_0.txt
        facebook_1.txt
        ...
        bio-SC-HT_96.txt
    """
    root = generated_root(model, method)

    paths = sorted(root.glob(f"{dataset}_*.txt"), key=natural_trial_sort_key)

    # Fallback for exact single-file names, if any result directory uses them.
    if not paths:
        exact = root / f"{dataset}.txt"
        if exact.exists():
            paths = [exact]

    if not paths:
        warnings.warn(
            f"No generated graphs found for dataset={dataset}, "
            f"model={model}, method={method}, root={root}"
        )

    if paths and len(paths) != EXPECTED_B:
        warnings.warn(
            f"Expected {EXPECTED_B} generated graphs for dataset={dataset}, "
            f"model={model}, method={method}, but found {len(paths)} in {root}"
        )

    return paths

# =========================================================
# METRICS
# =========================================================


def compute_metrics(G, metric_names: list[str] | None = None) -> dict[str, float]:
    """
    Lazily compute only the requested metrics.

    This intentionally avoids M.summarize_graph(G), because summarize_graph
    eagerly computes the whole default bundle. A smoke test can set
    METRIC_NAMES = ["gcc"].
    """
    if metric_names is None:
        metric_names = METRIC_NAMES

    cache: dict[str, dict[str, float]] = {}

    def get_k4() -> dict[str, float]:
        if "k4" not in cache:
            density, count = M.k_clique_density_and_count(G, 4, show_progress=False)
            cache["k4"] = {
                "k4_density": float(density),
                "k4_count": float(count),
            }
        return cache["k4"]

    metric_registry = {
        "n_nodes": lambda: float(G.number_of_nodes()),
        "n_edges": lambda: float(G.number_of_edges()),
        "n_triangles": lambda: float(M.unique_triangle_count(G)),
        "n_wedges": lambda: float(M.wedge_count(G)),

        # Operational definitions from metrics.py / binding repo alignment.
        "gcc": lambda: float(M.gcc(G)),
        "alcc": lambda: float(M.alcc(G)),

        # Exact clique metrics.
        "k4_count": lambda: get_k4()["k4_count"],
        "k4_density": lambda: get_k4()["k4_density"],

        # Exact higher-order clustering.
        "C3_global": lambda: float(M.higher_order_global_clustering(G, 3, show_progress=False)),
        "C3_avg_local": lambda: float(M.higher_order_average_local_clustering(G, 3, show_progress=False)),

        # Backward-compatible aliases, if needed.
        "k4": lambda: get_k4()["k4_count"],
        "C3": lambda: float(M.higher_order_global_clustering(G, 3, show_progress=False)),
    }

    unknown = [m for m in metric_names if m not in metric_registry]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. "
            f"Available metrics: {sorted(metric_registry.keys())}"
        )

    return {m: metric_registry[m]() for m in metric_names}

# =========================================================
# MONTE CARLO TEST
# =========================================================


def monte_carlo_test(x0: float, sims) -> dict[str, float | bool | str | int]:
    sims = np.asarray(sims, dtype=float)

    if len(sims) == 0:
        return {
            "real": float(x0),
            "mean_sim": np.nan,
            "std_sim": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "reject": False,
            "n_sims": 0,
            "note": "NO_SIM_DATA",
        }

    n_le = np.sum(sims <= x0)
    n_ge = np.sum(sims >= x0)

    # Two-sided empirical Monte Carlo p-value with plus-one correction.
    p = min(
        1.0,
        2 * min(
            (n_le + 1) / (len(sims) + 1),
            (n_ge + 1) / (len(sims) + 1),
        ),
    )

    ci_low, ci_high = np.percentile(sims, [2.5, 97.5])

    return {
        "real": float(x0),
        "mean_sim": float(np.mean(sims)),
        "std_sim": float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p),
        "reject": bool(p <= ALPHA),
        "n_sims": int(len(sims)),
    }

# =========================================================
# CACHING AND RESUMPTION LOGIC
# =========================================================


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _model_key_from_value(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value)
    if value in RESULT_DIRS or value in MODELS_DISPLAY:
        return value
    display_to_key = {v: k for k, v in MODELS_DISPLAY.items()}
    return display_to_key.get(value, value)


def _trial_key_from_trial_value(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def normalize_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Make older raw_metrics.csv files compatible with the resume logic."""
    if raw_df.empty:
        return raw_df

    raw_df = raw_df.copy()

    if "model_key" not in raw_df.columns:
        if "model" not in raw_df.columns:
            raise ValueError("Existing raw_metrics.csv has no model_key or model column.")
        raw_df["model_key"] = raw_df["model"].map(_model_key_from_value)

    if "trial_key" not in raw_df.columns:
        if "graph_file" in raw_df.columns:
            raw_df["trial_key"] = raw_df["graph_file"].map(trial_key_from_path)
        elif "trial" in raw_df.columns:
            raw_df["trial_key"] = raw_df["trial"].map(_trial_key_from_trial_value)
        else:
            raise ValueError(
                "Existing raw_metrics.csv has no trial_key, graph_file, or trial column."
            )

    raw_df["model_key"] = raw_df["model_key"].astype(str)
    raw_df["trial_key"] = raw_df["trial_key"].astype(str)

    return raw_df


def normalize_test_df(test_df: pd.DataFrame) -> pd.DataFrame:
    """Make older hypothesis_results.csv files compatible with the resume logic."""
    if test_df.empty:
        return test_df

    test_df = test_df.copy()

    if "model_key" not in test_df.columns:
        if "model" not in test_df.columns:
            raise ValueError("Existing hypothesis_results.csv has no model_key or model column.")
        test_df["model_key"] = test_df["model"].map(_model_key_from_value)

    test_df["model_key"] = test_df["model_key"].astype(str)
    return test_df


def load_existing_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    if OVERWRITE_RESULTS or not RESUME_FROM_CSV:
        return pd.DataFrame(), pd.DataFrame()

    raw_df = normalize_raw_df(_read_csv_if_exists(RAW_OUT))
    test_df = normalize_test_df(_read_csv_if_exists(TEST_OUT))

    if not raw_df.empty:
        raw_df = raw_df.drop_duplicates(RAW_KEY_COLS, keep="last")

    if not test_df.empty:
        test_df = test_df.drop_duplicates(TEST_KEY_COLS, keep="last")

    return raw_df, test_df


def save_checkpoint(raw_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if not raw_df.empty:
        raw_df = normalize_raw_df(raw_df).drop_duplicates(RAW_KEY_COLS, keep="last")

    if not test_df.empty:
        test_df = normalize_test_df(test_df).drop_duplicates(TEST_KEY_COLS, keep="last")

    raw_df.to_csv(RAW_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)


def raw_metric_columns_available(raw_df: pd.DataFrame) -> bool:
    return all(metric in raw_df.columns for metric in METRIC_NAMES)


def completed_trial_keys(raw_df: pd.DataFrame, dataset: str, model: str, method: str) -> set[str]:
    """
    Return trial_keys whose requested metric columns are already present and non-null.
    """
    if raw_df.empty or not raw_metric_columns_available(raw_df):
        return set()

    subset = raw_df[
        (raw_df["dataset"] == dataset)
        & (raw_df["model_key"] == model)
        & (raw_df["method"] == method)
    ]

    if subset.empty:
        return set()

    complete_mask = subset[METRIC_NAMES].notna().all(axis=1)
    return set(subset.loc[complete_mask, "trial_key"].astype(str))


def hypothesis_complete(
    test_df: pd.DataFrame,
    dataset: str,
    model: str,
    method: str,
    expected_n_sims: int,
) -> bool:
    """
    Check whether hypothesis_results.csv already has the requested metric tests.

    The n_sims check prevents a stale partial hypothesis row from being treated as
    complete after an interrupted run.
    """
    if test_df.empty:
        return False

    required_cols = set(TEST_KEY_COLS + ["real", "p_value", "reject", "n_sims"])
    if not required_cols.issubset(test_df.columns):
        return False

    subset = test_df[
        (test_df["dataset"] == dataset)
        & (test_df["model_key"] == model)
        & (test_df["method"] == method)
        & (test_df["metric"].isin(METRIC_NAMES))
    ]

    if set(subset["metric"]) != set(METRIC_NAMES):
        return False

    if not subset[["real", "p_value", "reject", "n_sims"]].notna().all().all():
        return False

    return bool((subset["n_sims"].astype(int) == expected_n_sims).all())


def remove_existing_hypothesis_rows(
    test_df: pd.DataFrame,
    dataset: str,
    model: str,
    method: str,
) -> pd.DataFrame:
    if test_df.empty:
        return test_df

    keep = ~(
        (test_df["dataset"] == dataset)
        & (test_df["model_key"] == model)
        & (test_df["method"] == method)
        & (test_df["metric"].isin(METRIC_NAMES))
    )

    return test_df[keep].copy()

# =========================================================
# MAIN PIPELINE
# =========================================================


def iter_ground_truth_files() -> list[Path]:
    if DATASETS is None:
        return sorted(GT_DIR.glob("*.txt"))
    return [GT_DIR / f"{dataset}.txt" for dataset in DATASETS]


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df, test_df = load_existing_results()

    for gt_path in tqdm(
        iter_ground_truth_files(),
        desc="Datasets",
        position=0,
        leave=True,
        **TQDM_KW,
    ):
        dataset = gt_path.stem

        G_real = None
        real_metrics = None

        def get_real_metrics() -> dict[str, float]:
            nonlocal G_real, real_metrics
            if real_metrics is None:
                G_real = M.load_graph(gt_path, show_progress=False)
                real_metrics = compute_metrics(G_real)
            return real_metrics

        for model in tqdm(
            MODELS,
            desc=f"Dataset={dataset} | Models",
            position=1,
            leave=False,
            **TQDM_KW,
        ):
            for method in tqdm(
                METHODS,
                desc=f"Model={model} | Methods",
                position=2,
                leave=False,
                **TQDM_KW,
            ):
                sim_paths = generated_graph_paths(dataset, model, method)
                expected_trial_keys = {trial_key_from_path(p) for p in sim_paths}
                expected_n_sims = len(sim_paths)

                done_trial_keys = completed_trial_keys(raw_df, dataset, model, method)
                raw_complete = expected_trial_keys.issubset(done_trial_keys)
                test_complete = hypothesis_complete(
                    test_df,
                    dataset,
                    model,
                    method,
                    expected_n_sims=expected_n_sims,
                )

                if raw_complete and test_complete:
                    continue

                missing_paths = [
                    p for p in sim_paths
                    if trial_key_from_path(p) not in done_trial_keys
                ]

                graph_bar = tqdm(
                    missing_paths,
                    desc=f"{dataset} | {model} | {method} | Missing graphs",
                    position=3,
                    leave=False,
                    **TQDM_KW,
                )

                graphs_since_checkpoint = 0

                for graph_path in graph_bar:
                    graph_bar.set_postfix_str(graph_path.name)

                    G_sim = M.load_graph(graph_path, show_progress=False)
                    stats = compute_metrics(G_sim)

                    row = {
                        "dataset": dataset,
                        "model": MODELS_DISPLAY.get(model, model),
                        "model_key": model,
                        "method": method,
                        "trial": trial_id_from_path(graph_path),
                        "trial_key": trial_key_from_path(graph_path),
                        "graph_file": str(graph_path),
                        **stats,
                    }

                    raw_df = pd.concat([raw_df, pd.DataFrame([row])], ignore_index=True)
                    raw_df = normalize_raw_df(raw_df).drop_duplicates(RAW_KEY_COLS, keep="last")

                    graphs_since_checkpoint += 1
                    if graphs_since_checkpoint >= CHECKPOINT_EVERY_N_GRAPHS:
                        save_checkpoint(raw_df, test_df)
                        graphs_since_checkpoint = 0

                # Rebuild the hypothesis rows from raw_df for this combo.
                combo_raw = raw_df[
                    (raw_df["dataset"] == dataset)
                    & (raw_df["model_key"] == model)
                    & (raw_df["method"] == method)
                    & (raw_df["trial_key"].astype(str).isin(expected_trial_keys))
                ].copy()

                if len(combo_raw) != expected_n_sims:
                    warnings.warn(
                        f"Only {len(combo_raw)} / {expected_n_sims} raw metric rows available "
                        f"for dataset={dataset}, model={model}, method={method}. "
                        "Hypothesis rows will be computed from available rows."
                    )

                real = get_real_metrics()
                new_test_rows = []

                for metric_name in METRIC_NAMES:
                    if metric_name not in combo_raw.columns:
                        sims = np.array([], dtype=float)
                    else:
                        sims = combo_raw[metric_name].dropna().to_numpy(dtype=float)

                    res = monte_carlo_test(real[metric_name], sims)
                    new_test_rows.append({
                        "dataset": dataset,
                        "model": MODELS_DISPLAY.get(model, model),
                        "model_key": model,
                        "method": method,
                        "metric": metric_name,
                        **res,
                    })

                test_df = remove_existing_hypothesis_rows(test_df, dataset, model, method)
                test_df = pd.concat([test_df, pd.DataFrame(new_test_rows)], ignore_index=True)
                test_df = normalize_test_df(test_df).drop_duplicates(TEST_KEY_COLS, keep="last")

                save_checkpoint(raw_df, test_df)

    return raw_df, test_df

# =========================================================
# VISUALIZATION
# =========================================================


def _ordered_pipeline_labels() -> tuple[dict[str, int], list[str]]:
    y_map: dict[str, int] = {}
    labels: list[str] = []
    idx = 0
    for model in MODELS:
        for method in METHODS:
            key = f"{model}+{method}"
            y_map[key] = idx
            labels.append(f"{MODELS_DISPLAY.get(model, model)}+{method}")
            idx += 1
    return y_map, labels


def plot_swimlane(raw_df: pd.DataFrame, test_df: pd.DataFrame, metric: str = "gcc", dataset: str | None = None) -> None:
    if metric not in raw_df.columns:
        warnings.warn(f"Metric {metric!r} not found in raw_df; skipping swimlane plot.")
        return

    if dataset is not None:
        raw_df = raw_df[raw_df["dataset"] == dataset]
        test_df = test_df[test_df["dataset"] == dataset]

    test_metric_df = test_df[test_df["metric"] == metric]
    y_map, labels = _ordered_pipeline_labels()

    plt.figure(figsize=(11, 6))

    for _, row in raw_df.iterrows():
        key = f"{row['model_key']}+{row['method']}"
        if key not in y_map:
            continue
        plt.scatter(row[metric], y_map[key], color="black", s=20, alpha=0.45)

    label_used = False
    for _, row in test_metric_df.iterrows():
        key = f"{row['model_key']}+{row['method']}"
        if key not in y_map:
            continue
        y = y_map[key]
        plt.plot([row["ci_low"], row["ci_high"]], [y, y], linewidth=3)
        plt.scatter(
            row["real"],
            y,
            marker="*",
            s=180,
            color="black",
            label="Ground truth" if not label_used else "",
        )
        label_used = True

    title_prefix = f"{dataset}: " if dataset else ""
    plt.yticks(list(y_map.values()), labels)
    plt.xlabel(metric.upper())
    plt.ylabel("Pipeline (Model + Method)")
    plt.title(f"{title_prefix}{metric.upper()} - Monte Carlo Test with Simulation Spread")
    plt.grid(alpha=0.25)
    if label_used:
        plt.legend()
    plt.tight_layout()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_part = f"{dataset}_" if dataset else ""
    out_path = ANALYSIS_DIR / f"{dataset_part}{metric}_swimlane.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_percentile_band(test_df: pd.DataFrame, metric: str = "gcc", dataset: str | None = None) -> None:
    if dataset is not None:
        test_df = test_df[test_df["dataset"] == dataset]

    subset = test_df[test_df["metric"] == metric].copy()
    if subset.empty:
        warnings.warn(f"No rows to plot for dataset={dataset}, metric={metric}")
        return

    method_order = {method: i for i, method in enumerate(METHODS)}
    model_order = {model: i for i, model in enumerate(MODELS)}
    subset["_model_order"] = subset["model_key"].map(model_order).fillna(10**6)
    subset["_method_order"] = subset["method"].map(method_order).fillna(10**6)
    subset = subset.sort_values(["_model_order", "_method_order"])

    subset["label"] = subset.apply(lambda r: f"{r['model']}+{r['method']}", axis=1)

    plt.figure(figsize=(10, 5))
    y = np.arange(len(subset))

    plt.hlines(y, subset["ci_low"], subset["ci_high"], linewidth=6)
    plt.scatter(subset["real"], y, marker="*", s=150, color="black")

    title_prefix = f"{dataset}: " if dataset else ""
    plt.yticks(y, subset["label"])
    plt.xlabel(metric.upper())
    plt.title(f"{title_prefix}{metric.upper()} Percentile Bands (2.5%–97.5%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_part = f"{dataset}_" if dataset else ""
    out_path = ANALYSIS_DIR / f"{dataset_part}{metric}_percentile_band.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    raw_df, test_df = run_analysis()
    save_checkpoint(raw_df, test_df)

    print("\n===== FINAL HYPOTHESIS TABLE =====")
    display_cols = [
        "dataset",
        "model",
        "method",
        "metric",
        "real",
        "ci_low",
        "ci_high",
        "p_value",
        "reject",
        "n_sims",
    ]
    available_display_cols = [c for c in display_cols if c in test_df.columns]
    print(test_df[available_display_cols] if available_display_cols else test_df)

    if not test_df.empty and "dataset" in test_df.columns:
        for dataset in sorted(test_df["dataset"].dropna().unique()):
            for metric in METRIC_NAMES:
                plot_swimlane(raw_df, test_df, metric=metric, dataset=dataset)
                plot_percentile_band(test_df, metric=metric, dataset=dataset)
