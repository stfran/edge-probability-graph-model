import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from experiment_pipeline import (
    MODELS,
    METHODS,
    MODELS_DISPLAY,
    METRIC_NAMES,
    RAW_OUT,
    TEST_OUT,
    ANALYSIS_DIR,
)

PLOTS_DIR = ANALYSIS_DIR / "plots"


# =========================================================
# HELPERS
# =========================================================

def _filter_dataset(df: pd.DataFrame, dataset: str | None) -> pd.DataFrame:
    if dataset is None or df.empty or "dataset" not in df.columns:
        return df
    return df[df["dataset"] == dataset].copy()


def _active_pipeline_order(
    raw_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metric: str,
) -> list[tuple[str, str]]:
    """
    Return only the (model_key, method) pairs that actually have data.

    This avoids showing empty pipeline rows for runs that have not started
    or have not produced any rows yet.
    """
    active = set()

    if not raw_df.empty and metric in raw_df.columns:
        raw_subset = raw_df[raw_df[metric].notna()]
        for _, row in raw_subset.iterrows():
            active.add((str(row["model_key"]), str(row["method"])))

    if not test_df.empty:
        test_subset = test_df[test_df["metric"] == metric]
        for _, row in test_subset.iterrows():
            active.add((str(row["model_key"]), str(row["method"])))

    ordered = []
    for model in MODELS:
        for method in METHODS:
            if (model, method) in active:
                ordered.append((model, method))

    return ordered


def _ground_truth_value(test_metric_df: pd.DataFrame) -> float | None:
    """
    Extract one ground-truth value for the dataset/metric.

    In principle, 'real' should be the same across all model/method rows for a
    given dataset+metric. If multiple values are present, warn and use the first.
    """
    if test_metric_df.empty or "real" not in test_metric_df.columns:
        return None

    vals = test_metric_df["real"].dropna().astype(float).unique()
    if len(vals) == 0:
        return None

    if len(vals) > 1:
        warnings.warn(
            f"Multiple distinct ground-truth 'real' values found for one dataset/metric: {vals}. "
            f"Using the first value {vals[0]}."
        )

    return float(vals[0])


# =========================================================
# VISUALIZATION
# =========================================================

def plot_monte_carlo_summary(
    raw_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metric: str = "gcc",
    dataset: str | None = None,
) -> None:
    """
    Single consolidated plot:
      - gray/black dots = generated graph metric values (Monte Carlo samples)
      - red horizontal box = empirical 95% interval (2.5%–97.5%)
      - black median tick inside the box
      - green vertical line = ground-truth metric value

    Only active pipelines are shown.
    """
    if raw_df.empty and test_df.empty:
        warnings.warn("Both raw_df and test_df are empty; skipping plot.")
        return

    raw_df = _filter_dataset(raw_df, dataset)
    test_df = _filter_dataset(test_df, dataset)

    if raw_df.empty and test_df.empty:
        warnings.warn(f"No rows to plot for dataset={dataset}, metric={metric}")
        return

    if not raw_df.empty and metric in raw_df.columns:
        raw_metric_df = raw_df[raw_df[metric].notna()].copy()
    else:
        raw_metric_df = pd.DataFrame()

    test_metric_df = test_df[test_df["metric"] == metric].copy()
    active_pipelines = _active_pipeline_order(raw_metric_df, test_metric_df, metric)

    if not active_pipelines:
        warnings.warn(f"No active pipelines to plot for dataset={dataset}, metric={metric}")
        return

    y_map = {}
    labels = []
    for idx, (model, method) in enumerate(active_pipelines):
        key = f"{model}+{method}"
        y_map[key] = idx
        labels.append(f"{MODELS_DISPLAY.get(model, model)}+{method}")

    fig_height = max(4.5, 0.75 * len(active_pipelines) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    # Visual tuning
    ci_box_height = 0.18
    median_tick_halfheight = 0.13
    dot_jitter = 0.035

    # -----------------------------------------------------
    # Plot generated graph samples as dots with slight jitter
    # -----------------------------------------------------
    if not raw_metric_df.empty:
        rng = np.random.default_rng(0)  # deterministic jitter
        for _, row in raw_metric_df.iterrows():
            key = f"{row['model_key']}+{row['method']}"
            if key not in y_map:
                continue

            y = y_map[key] + rng.uniform(-dot_jitter, dot_jitter)

            ax.scatter(
                row[metric],
                y,
                color="black",
                s=22,
                alpha=0.35,
                zorder=3,
            )

    # -----------------------------------------------------
    # Plot CI as a horizontal rectangle + median tick
    # -----------------------------------------------------
    for model, method in active_pipelines:
        key = f"{model}+{method}"
        y = y_map[key]

        row = test_metric_df[
            (test_metric_df["model_key"] == model)
            & (test_metric_df["method"] == method)
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")

        if pd.notna(ci_low) and pd.notna(ci_high):
            width = ci_high - ci_low
            if width < 0:
                continue

            rect = Rectangle(
                (ci_low, y - ci_box_height / 2),
                width,
                ci_box_height,
                facecolor="green",
                edgecolor="green",
                alpha=0.28,
                linewidth=2.0,
                zorder=2,
            )
            ax.add_patch(rect)

        # Median tick from raw samples for this pipeline
        if not raw_metric_df.empty:
            sims = raw_metric_df[
                (raw_metric_df["model_key"] == model)
                & (raw_metric_df["method"] == method)
            ][metric].dropna()

            if not sims.empty:
                med = float(np.median(sims))
                ax.vlines(
                    med,
                    y - median_tick_halfheight,
                    y + median_tick_halfheight,
                    color="black",
                    linewidth=2.5,
                    zorder=4,
                )

    # -----------------------------------------------------
    # Plot one vertical line for ground truth
    # -----------------------------------------------------
    real_x = _ground_truth_value(test_metric_df)
    if real_x is not None:
        ax.axvline(
            x=real_x,
            color="green",
            linewidth=2.5,
            zorder=1,
        )

    # -----------------------------------------------------
    # Labels / legend / cosmetics
    # -----------------------------------------------------
    title_prefix = f"{dataset}: " if dataset else ""
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Pipeline (Model + Method)")
    ax.set_title(f"{title_prefix}{metric.upper()} - Monte Carlo Summary")
    ax.grid(alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="black",
            markersize=6,
            alpha=0.35,
            label="Generated graphs (Monte Carlo samples)",
        ),
        Rectangle(
            (0, 0), 1, 1,
            facecolor="red",
            edgecolor="red",
            alpha=0.28,
            label="95% empirical interval (2.5%–97.5%)",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.5,
            label="Median of generated graphs",
        ),
    ]

    if real_x is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="green",
                linewidth=2.5,
                label="Ground truth",
            )
        )

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_part = f"{dataset}_" if dataset else ""
    out_path = PLOTS_DIR / f"{dataset_part}{metric}_monte_carlo_summary.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_OUT)
    test_df = pd.read_csv(TEST_OUT)

    if not test_df.empty and "dataset" in test_df.columns:
        for dataset in sorted(test_df["dataset"].dropna().unique()):
            for metric in METRIC_NAMES:
                plot_monte_carlo_summary(
                    raw_df,
                    test_df,
                    metric=metric,
                    dataset=dataset,
                )