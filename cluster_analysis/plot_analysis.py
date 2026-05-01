import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

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
    Return only active (model_key, method) pairs that actually have data.
    Keep the order given by MODELS x METHODS.
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
    if test_metric_df.empty or "real" not in test_metric_df.columns:
        return None

    vals = test_metric_df["real"].dropna().astype(float).unique()
    if len(vals) == 0:
        return None

    if len(vals) > 1:
        warnings.warn(
            f"Multiple distinct ground-truth values found for one dataset/metric: {vals}. "
            f"Using the first value {vals[0]}."
        )

    return float(vals[0])


def _build_swimlane_layout(
    active_pipelines: list[tuple[str, str]],
) -> tuple[dict[str, float], list[float], list[str], list[dict], list[float]]:
    """
    Create y-positions with gaps between model groups.

    Returns
    -------
    y_map:
        "MODEL+METHOD" -> y position
    y_ticks:
        y positions for method tick labels
    y_ticklabels:
        method labels only
    groups:
        list of dicts containing model group metadata
    separators:
        y values at which to draw separator lines between model groups
    """
    y_map = {}
    y_ticks = []
    y_ticklabels = []
    groups = []

    group_gap = 0.75
    current_y = 0.0

    for model in MODELS:
        present_methods = [m for m in METHODS if (model, m) in active_pipelines]
        if not present_methods:
            continue

        start_y = current_y

        for method in present_methods:
            key = f"{model}+{method}"
            y_map[key] = current_y
            y_ticks.append(current_y)
            y_ticklabels.append(method)
            current_y += 1.0

        end_y = current_y - 1.0

        groups.append(
            {
                "model": model,
                "start_y": start_y,
                "end_y": end_y,
                "center_y": (start_y + end_y) / 2.0,
            }
        )

        current_y += group_gap

    separators = []
    for i in range(len(groups) - 1):
        separators.append((groups[i]["end_y"] + groups[i + 1]["start_y"]) / 2.0)

    return y_map, y_ticks, y_ticklabels, groups, separators

def _display_model_label(model: str) -> str:
    label = MODELS_DISPLAY.get(model, model)

    # Manual line breaks for long labels.
    replacements = {
        "Stochastic Block Model": "Stochastic\nBlock Model",
        "Erdős–Rényi": "Erdős–Rényi",
    }

    return replacements.get(label, label)


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
    Combined Monte Carlo summary plot:
      - dots = generated-graph metric values
      - shaded box = empirical 95% interval (2.5%-97.5%)
      - tall black tick = median of generated graphs
      - green vertical line = ground truth

    Visual design:
      - method labels on y-axis
      - model names rotated 90 degrees and grouped like swimlanes
      - dark separator lines between model groups
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

    y_map, y_ticks, y_ticklabels, groups, separators = _build_swimlane_layout(active_pipelines)

    # Squish vertically by about 30% compared to the older taller layout.
    fig_height = max(4.0, 0.50 * len(y_ticks) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    # Leave more room at left for rotated model labels.
    fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.12)

    # Visual tuning
    ci_box_height = 0.24
    median_tick_halfheight = 0.34   # taller median marker
    dot_jitter = 0.04

    # -----------------------------------------------------
    # Generated graph samples as dots with slight jitter
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
                alpha=0.30,
                zorder=3,
            )

    # -----------------------------------------------------
    # CI box + median tick
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
            if width >= 0:
                rect = Rectangle(
                    (ci_low, y - ci_box_height / 2),
                    width,
                    ci_box_height,
                    facecolor="#7BC67B",
                    edgecolor="#4C9A4C",
                    alpha=0.45,
                    linewidth=1.8,
                    zorder=2,
                )
                ax.add_patch(rect)

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
                    color="#5D645DA6",        
                    linewidth=3.2,
                    linestyles="-",
                    zorder=5,
                )

    # -----------------------------------------------------
    # Ground-truth vertical line
    # -----------------------------------------------------
    real_x = _ground_truth_value(test_metric_df)
    if real_x is not None:
        ax.axvline(
            x=real_x,
            color="green",
            linewidth=2.2,
            zorder=1,
        )

        ax.annotate(
            "<-Ground truth",
            xy=(real_x, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(40, 12),
            textcoords="offset points",
            rotation=0,
            va="top",
            ha="center",
            color="black",
            fontsize=9,
            annotation_clip=False,
        )

    # -----------------------------------------------------
    # Dark separator lines between model groups
    # -----------------------------------------------------
    sep_trans = blended_transform_factory(ax.transAxes, ax.transData)

    for sep_y in separators:
        ax.plot(
            [-0.28, 1.0],
            [sep_y, sep_y],
            transform=sep_trans,
            color="black",
            linewidth=1.4,
            alpha=0.9,
            clip_on=False,
            zorder=0,
        )

    # -----------------------------------------------------
    # Rotated model labels in the left margin
    # -----------------------------------------------------
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for group in groups:
        ax.text(
            -0.14,
            group["center_y"],
            _display_model_label(group["model"]),
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            transform=trans,
        )

    # -----------------------------------------------------
    # Axes / legend / title
    # -----------------------------------------------------
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)

    ax.set_xlabel(metric.upper())
    ax.set_ylabel("")

    if dataset is not None:
        ax.set_title(
            f"Monte Carlo Significance Test Summary on {metric.upper()} with dataset {dataset}"
        )
    else:
        ax.set_title(f"Monte Carlo Significance Test Summary on {metric.upper()}")

    ax.grid(alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="black",
            markersize=6,
            alpha=0.30,
            label="Generated graphs (Monte Carlo samples)",
        ),
        Rectangle(
            (0, 0), 1, 1,
            facecolor="#7BC67B",
            edgecolor="#4C9A4C",
            alpha=0.45,
            label="95% empirical interval (2.5%–97.5%)",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.6,
            label="Median of generated graphs",
        ),
    ]

    if real_x is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="green",
                linewidth=2.0,
                label="Ground truth",
            )
        )

    # ax.legend(handles=legend_handles, loc="lower left")
    # instead of legend, we'll describe elements with annotations and in the caption

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_part = f"{dataset}_" if dataset else ""
    out_path = PLOTS_DIR / f"{dataset_part}{metric}_monte_carlo_summary.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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