from __future__ import annotations

from pathlib import Path
import re
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import metrics as M

# =========================================================
# CONFIG
# =========================================================

# Use cached raw_metrics.csv / hypothesis_results.csv / ground_truth_metrics.csv
# when available.
RESUME_FROM_CSV = True
OVERWRITE_RESULTS = False

# Save after this many generated graphs. Set to 1 for best interruption safety.
CHECKPOINT_EVERY_N_GRAPHS = 1

# Show a progress bar over metric names, e.g., 1/6 metrics done.
# Ground-truth metric progress is always useful; generated-graph metric progress
# is controlled separately below.
SHOW_METRIC_PROGRESS = True
SHOW_GENERATED_METRIC_PROGRESS = False

# Directories
BASE_DIR = Path(__file__).resolve().parent
GT_DIR = BASE_DIR.parent / "data" / "gt_txt"
EDGE_IND_RESULTS_DIR = BASE_DIR / "results" / "data"
BIND_RESULTS_DIR = BASE_DIR / "results" / "gen_res"
ANALYSIS_DIR = BASE_DIR / "analysis"

RAW_OUT = ANALYSIS_DIR / "raw_metrics.csv"
TEST_OUT = ANALYSIS_DIR / "hypothesis_results.csv"
GT_METRICS_OUT = ANALYSIS_DIR / "ground_truth_metrics.csv"

# HYPOTHESIS TEST PARAMETERS
EXPECTED_B = 100  # warn if a generated-results directory has a different count
ALPHA = 0.05

# RUN CONFIGURATION
# None runs all datasets.
# To run subsets, use any of:
# ["facebook", "bio-CE-PG", "bio-SC-HT", "hamsterster", "polblogs", "web-spam"]
# DATASETS = ["facebook"]
DATASETS = None

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
#    "k4_count",
    "k4_density",
    "C3_global",
#    "C3_avg_local",
]

TQDM_KW = {
    "dynamic_ncols": True,
    "mininterval": 0.5,
}

# RESULTS COLUMNS
RAW_KEY_COLS = ["dataset", "model_key", "method", "trial_key"]
TEST_KEY_COLS = ["dataset", "model_key", "method", "metric"]
GT_KEY_COLS = ["dataset", "metric"]

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


def compute_metrics(
    G,
    metric_names: list[str] | tuple[str, ...] | None = None,
    *,
    desc: str = "Computing metrics",
    show_progress: bool = SHOW_METRIC_PROGRESS,
    progress_position: int = 4,
) -> dict[str, float]:
    """
    Lazily compute only the requested metrics.

    The progress bar is over metric names, not graph-loading lines or graph
    nodes. This makes expensive metric bundles easier to track.
    """
    if metric_names is None:
        metric_names = METRIC_NAMES
    metric_names = list(metric_names)

    cache = {}

    def get_k4():
        if "k4" not in cache:
            density, count = M.k_clique_density_and_count(G, 4)
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

        # Operational definitions from metrics.py / original binding code.
        "gcc": lambda: float(M.gcc(G)),
        "alcc": lambda: float(M.alcc(G)),

        # Exact 4-clique metrics.
        "k4_count": lambda: get_k4()["k4_count"],
        "k4_density": lambda: get_k4()["k4_density"],

        # Exact higher-order clustering metrics.
        "C3_global": lambda: float(M.higher_order_global_clustering(G, 3)),
        "C3_avg_local": lambda: float(M.higher_order_average_local_clustering(G, 3)),

        # Backward-compatible aliases.
        "k4": lambda: get_k4()["k4_count"],
        "C3": lambda: float(M.higher_order_global_clustering(G, 3)),
    }

    unknown = [m for m in metric_names if m not in metric_registry]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. "
            f"Available metrics: {sorted(metric_registry.keys())}"
        )

    iterator = tqdm(
        metric_names,
        desc=desc,
        position=progress_position,
        leave=False,
        disable=not show_progress,
        **TQDM_KW,
    )

    out: dict[str, float] = {}
    for metric_name in iterator:
        iterator.set_postfix_str(metric_name)
        out[metric_name] = metric_registry[metric_name]()

    return out

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


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Write a CSV via a temporary file then rename it into place.

    This reduces the chance of leaving a partially written CSV if the process is
    interrupted during checkpointing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp, index=False)

    tmp_path.replace(path)


def _model_key_from_value(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value)

    if value in MODELS_DISPLAY:
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

    _atomic_write_csv(raw_df, RAW_OUT)
    _atomic_write_csv(test_df, TEST_OUT)


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

    The n_sims check prevents a stale partial hypothesis row from being treated
    as complete after an interrupted run.
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


def upsert_raw_row(raw_df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """
    Insert or update one raw metric row while preserving older metric columns
    that are not present in the new row.

    This is useful when you first run METRIC_NAMES=["gcc"] and later rerun with
    additional metrics.
    """
    if raw_df.empty:
        return pd.DataFrame([row])

    key_mask = pd.Series(True, index=raw_df.index)
    for col in RAW_KEY_COLS:
        key_mask &= raw_df[col].astype(str) == str(row[col])

    if not key_mask.any():
        return pd.concat([raw_df, pd.DataFrame([row])], ignore_index=True)

    idx = raw_df.index[key_mask][0]

    for col, val in row.items():
        if col not in raw_df.columns:
            raw_df[col] = np.nan
        raw_df.at[idx, col] = val

    return raw_df

# ---------------------------------------------------------------------------
# Ground-truth metric cache
# ---------------------------------------------------------------------------


def load_ground_truth_metrics() -> pd.DataFrame:
    if OVERWRITE_RESULTS or not RESUME_FROM_CSV or not GT_METRICS_OUT.exists():
        return pd.DataFrame(columns=["dataset", "metric", "value"])

    gt_df = _read_csv_if_exists(GT_METRICS_OUT)
    if gt_df.empty:
        return pd.DataFrame(columns=["dataset", "metric", "value"])

    required = {"dataset", "metric", "value"}
    if not required.issubset(gt_df.columns):
        raise ValueError(
            f"Existing {GT_METRICS_OUT} must contain columns {sorted(required)}"
        )

    return gt_df.drop_duplicates(GT_KEY_COLS, keep="last")


def save_ground_truth_metrics(gt_df: pd.DataFrame) -> None:
    gt_df = gt_df.drop_duplicates(GT_KEY_COLS, keep="last")
    _atomic_write_csv(gt_df, GT_METRICS_OUT)


def get_cached_ground_truth_metrics(
    gt_df: pd.DataFrame,
    dataset: str,
    metric_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, float], set[str]]:
    if metric_names is None:
        metric_names = METRIC_NAMES
    metric_names = list(metric_names)

    if gt_df.empty:
        return {}, set(metric_names)

    subset = gt_df[
        (gt_df["dataset"] == dataset)
        & (gt_df["metric"].isin(metric_names))
    ]

    cached = {
        row["metric"]: float(row["value"])
        for _, row in subset.iterrows()
        if pd.notna(row["value"])
    }

    missing = set(metric_names) - set(cached.keys())
    return cached, missing


def get_ground_truth_metrics(
    dataset: str,
    gt_path: Path,
    gt_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Return ground-truth metrics for this dataset, computing only missing metrics.

    This prevents recomputing expensive deterministic ground-truth metrics on
    every run.
    """
    cached, missing = get_cached_ground_truth_metrics(gt_df, dataset, METRIC_NAMES)

    if not missing:
        tqdm.write(f"[cache hit] ground-truth metrics: {dataset}")
        return cached, gt_df

    missing_ordered = [m for m in METRIC_NAMES if m in missing]
    tqdm.write(
        f"[cache miss] ground-truth metrics: {dataset}; "
        f"computing {missing_ordered}"
    )

    G_real = M.load_graph(gt_path, show_progress=False)

    newly_computed = compute_metrics(
        G_real,
        metric_names=missing_ordered,
        desc=f"{dataset} | ground truth metrics",
        show_progress=True,
        progress_position=4,
    )

    new_rows = [
        {"dataset": dataset, "metric": metric_name, "value": value}
        for metric_name, value in newly_computed.items()
    ]

    if new_rows:
        gt_df = pd.concat([gt_df, pd.DataFrame(new_rows)], ignore_index=True)
        gt_df = gt_df.drop_duplicates(GT_KEY_COLS, keep="last")
        save_ground_truth_metrics(gt_df)

    # Return a complete mapping in METRIC_NAMES order.
    combined = dict(cached)
    combined.update(newly_computed)
    return {m: combined[m] for m in METRIC_NAMES}, gt_df

# =========================================================
# MAIN PIPELINE
# =========================================================


def iter_ground_truth_files() -> list[Path]:
    if DATASETS is None:
        return sorted(GT_DIR.glob("*.txt"))
    return [GT_DIR / f"{dataset}.txt" for dataset in DATASETS]


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df, test_df = load_existing_results()
    gt_metrics_df = load_ground_truth_metrics()

    for gt_path in tqdm(
        iter_ground_truth_files(),
        desc="Datasets",
        position=0,
        leave=True,
        **TQDM_KW,
    ):
        dataset = gt_path.stem

        real_metrics, gt_metrics_df = get_ground_truth_metrics(
            dataset=dataset,
            gt_path=gt_path,
            gt_df=gt_metrics_df,
        )

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
                    stats = compute_metrics(
                        G_sim,
                        desc=(
                            f"{dataset} | {model} | {method} | "
                            f"trial {trial_key_from_path(graph_path)}"
                        ),
                        show_progress=SHOW_GENERATED_METRIC_PROGRESS,
                        progress_position=4,
                    )

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

                    raw_df = upsert_raw_row(raw_df, row)
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

                new_test_rows = []

                for metric_name in METRIC_NAMES:
                    if metric_name not in combo_raw.columns:
                        sims = np.array([], dtype=float)
                    else:
                        sims = combo_raw[metric_name].dropna().to_numpy(dtype=float)

                    res = monte_carlo_test(real_metrics[metric_name], sims)
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

    
