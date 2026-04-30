import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings


from experiment_pipeline import MODELS, METHODS, MODELS_DISPLAY, METRIC_NAMES, RAW_OUT, TEST_OUT, ANALYSIS_DIR

PLOTS_DIR = ANALYSIS_DIR / "plots"

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

def plot_swimlane(
    raw_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metric: str = "gcc",
    dataset: str | None = None,
) -> None:
    if raw_df.empty or metric not in raw_df.columns:
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

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_part = f"{dataset}_" if dataset else ""
    out_path = PLOTS_DIR / f"{dataset_part}{metric}_swimlane.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_percentile_band(
    test_df: pd.DataFrame,
    metric: str = "gcc",
    dataset: str | None = None,
) -> None:
    if test_df.empty:
        warnings.warn("test_df is empty; skipping percentile-band plot.")
        return

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

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_part = f"{dataset}_" if dataset else ""
    out_path = PLOTS_DIR / f"{dataset_part}{metric}_percentile_band.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":

    raw_df = pd.read_csv(RAW_OUT)
    test_df = pd.read_csv(TEST_OUT)

    if not test_df.empty and "dataset" in test_df.columns:
        for dataset in sorted(test_df["dataset"].dropna().unique()):
            for metric in METRIC_NAMES:
                plot_swimlane(raw_df, test_df, metric=metric, dataset=dataset)
                plot_percentile_band(test_df, metric=metric, dataset=dataset)