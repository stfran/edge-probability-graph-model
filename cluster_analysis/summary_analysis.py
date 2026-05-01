#!/usr/bin/env python3
"""
summary_analysis.py

Create consolidated hypothesis-test summaries from analysis/hypothesis_results.csv.

Outputs:
  analysis/summary/heatmap.pdf
  analysis/summary/aggregate_table.tex
  analysis/summary/full_numerical_table.tex

Expected input columns:
  dataset, model, model_key, method, metric, real, mean_sim, std_sim,
  ci_low, ci_high, p_value, reject, n_sims
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ANALYSIS_DIR = Path("analysis")
INPUT_CSV = ANALYSIS_DIR / "hypothesis_results.csv"
OUT_DIR = ANALYSIS_DIR / "summary"

HEATMAP_OUT = OUT_DIR / "heatmap.pdf"
AGG_TABLE_OUT = OUT_DIR / "aggregate_table.tex"
FULL_TABLE_OUT = OUT_DIR / "full_numerical_table.tex"


# ---------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------

DATASET_ORDER = [
    "hamsterster",
    "facebook",
    "polblogs",
    "web-spam",
    "bio-CE-PG",
    "bio-SC-HT",
]

MODEL_ORDER = ["ER", "CL", "SB", "SBM", "KR"]
METHOD_ORDER = ["EDGEIND", "LOCLBDG", "PARABDG"]

# Main-paper metrics. We intentionally omit k4_count because density is
# more comparable across datasets.
METRIC_ORDER = [
    "gcc",
    "alcc",
    "k4_density",
    "C3_global"
]

MODEL_LABELS = {
    "ER": "ER",
    "CL": "CL",
    "SB": "SB",
    "SBM": "SB",
    "KR": "KR",
}

METHOD_LABELS = {
    "EDGEIND": "Edge-independent",
    "LOCLBDG": "Local binding",
    "PARABDG": "Parallel binding",
}

TABLE_METHOD_LABELS = {
    "EDGEIND": "EDGEIND",
    "LOCLBDG": "LOCLBDG",
    "PARABDG": "PARABDG",
}

METRIC_LABELS = {
    "gcc": "GCC",
    "alcc": "ALCC",
#    "k4_count": r"$K_4$ count",
    "k4_density": r"$K_4$-density",
    "C3_global": r"$C_3$",
#    "C3_avg_local": r"$C_3$ avg. local",
}

LATEX_METRIC_LABELS = {
    "gcc": "GCC",
    "alcc": "ALCC",
#    "k4_count": r"$K_4$ count",
    "k4_density": r"$K_4$-density",
    "C3_global": r"$C_3$",
#    "C3_avg_local": r"$C_3$ avg. local",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def latex_escape(value: object) -> str:
    """Escape a value for use in LaTeX table cells."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def ordered_present(values: pd.Series, preferred_order: list[str]) -> list[str]:
    """Return preferred_order entries that are present, then any extras alphabetically."""
    present = set(values.dropna().astype(str).unique())
    ordered = [x for x in preferred_order if x in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def safe_log10_ratio(mean_sim: float, real: float) -> float:
    """
    Signed log10 ratio used for heatmap color.

    Returns:
      log10(mean_sim / real), when both are positive.
      NaN otherwise.

    In this study, most metrics should be nonnegative. If real or mean_sim
    is zero, the ratio is not meaningful for a log-ratio heatmap.
    """
    if pd.isna(mean_sim) or pd.isna(real):
        return np.nan
    if mean_sim <= 0 or real <= 0:
        return np.nan
    return math.log10(mean_sim / real)

def safe_ratio(mean_sim: float, real: float) -> float:
    """
    Return the raw ratio mean_sim / real for display.
    Returns NaN if undefined.

    For our graph metrics, values should be nonnegative. A ratio near 1
    means the generated mean is close to the ground truth.
    """
    if pd.isna(mean_sim) or pd.isna(real):
        return np.nan
    if real <= 0:
        return np.nan
    return float(mean_sim) / float(real)

def safe_standardized_deviation(mean_sim: float, real: float, std_sim: float) -> float:
    """
    Return standardized deviation (mean_sim - real) / std_sim.

    Interpretation:
      0    -> generated mean matches ground truth
      < 0  -> generated mean is below ground truth
      > 0  -> generated mean is above ground truth

    Returns NaN if undefined.
    """
    if pd.isna(mean_sim) or pd.isna(real) or pd.isna(std_sim):
        return np.nan

    mean_sim = float(mean_sim)
    real = float(real)
    std_sim = float(std_sim)

    if std_sim <= 0:
        # If the generator is effectively deterministic:
        # exact match -> 0, otherwise undefined/very large.
        if np.isclose(mean_sim, real):
            return 0.0
        return np.nan

    return (mean_sim - real) / std_sim


def fmt_num(x: float, digits: int = 3) -> str:
    """Compact numeric formatter for tables."""
    if pd.isna(x):
        return "--"

    x = float(x)
    abs_x = abs(x)

    if abs_x == 0:
        return "0"

    if abs_x < 1e-3 or abs_x >= 1e4:
        return f"{x:.{digits}e}"

    return f"{x:.{digits}g}"


def fmt_pvalue(p: float) -> str:
    if pd.isna(p):
        return "--"
    p = float(p)
    if p < 0.001:
        return r"$<10^{-3}$"
    return f"{p:.3f}"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, filter, and add derived columns."""
    required = {
        "dataset",
        "model",
        "model_key",
        "method",
        "metric",
        "real",
        "mean_sim",
        "std_sim",
        "ci_low",
        "ci_high",
        "p_value",
        "reject",
        "n_sims",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

    df = df.copy()

    if df["reject"].dtype != bool:
        df["reject"] = df["reject"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"]
        )

    df_main = df[df["metric"].isin(METRIC_ORDER)].copy()

    df_main["display_value"] = [
        safe_standardized_deviation(m, r, s)
        for m, r, s in zip(df_main["mean_sim"], df_main["real"], df_main["std_sim"])
    ]

    df_main["plausible"] = ~df_main["reject"]

    return df_main


# ---------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------

def make_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """
    Consolidated heatmap with:
      - three method panels,
      - rows grouped by dataset,
      - model labels on the y-axis,
      - dataset labels in the left margin,
      - horizontal separator lines between dataset blocks,
        extended into the left margin on the first panel.
    """
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.colors import TwoSlopeNorm

    datasets = ordered_present(df["dataset"], DATASET_ORDER)
    methods = ordered_present(df["method"], METHOD_ORDER)
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"])]

    # Stable model order
    preferred_models = ["ER", "CL", "SBM", "SB", "KR"]
    raw_models = ordered_present(df["model_key"], preferred_models)

    # Collapse SB/SBM aliases if both appear
    models = []
    seen_family = set()
    for m in raw_models:
        fam = "SB" if m in {"SB", "SBM"} else m
        if fam not in seen_family:
            models.append(m)
            seen_family.add(fam)

    if not datasets or not methods or not metrics or not models:
        raise ValueError("No datasets, methods, metrics, or models available for heatmap.")

    # ------------------------------------------------------------
    # Build row layout with NO vertical gap.
    # ------------------------------------------------------------
    row_items = []
    dataset_blocks = []
    current_row = 0

    for dataset in datasets:
        block_models = []
        for model in models:
            has_data = not df[
                (df["dataset"] == dataset) & (df["model_key"] == model)
            ].empty
            if has_data:
                block_models.append(model)

        if not block_models:
            continue

        start_row = current_row
        for model in block_models:
            row_items.append((dataset, model))
            current_row += 1
        end_row = current_row - 1

        dataset_blocks.append(
            {
                "dataset": dataset,
                "start_row": start_row,
                "end_row": end_row,
                "center_row": (start_row + end_row) / 2.0,
            }
        )

    n_rows = len(row_items)
    n_cols = len(metrics)
    n_methods = len(methods)

    fig_width = max(8.8, 2.65 * n_methods)
    fig_height = max(6.0, 0.42 * n_rows + 1.8)

    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(fig_width, fig_height),
        sharey=True,
        constrained_layout=False,
    )

    if n_methods == 1:
        axes = [axes]

    # Raw ratio scale:
    #   1.0 = perfect match (white-ish),
    #   <1.0 = underestimation (blue),
    #   >1.0 = overestimation (red).
    #
    # We clip to [0, 2] for now. Values outside that range are shown
    # via colorbar extensions.
    vmin, vcenter, vmax = -3.0, 0.0, 3.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = "coolwarm"
    last_im = None

    for ax_idx, (ax, method) in enumerate(zip(axes, methods)):
        mat = np.full((n_rows, n_cols), np.nan)
        plausible = np.full((n_rows, n_cols), False)
        observed = np.full((n_rows, n_cols), False)

        for i, (dataset, model) in enumerate(row_items):
            for j, metric in enumerate(metrics):
                rows = df[
                    (df["dataset"] == dataset)
                    & (df["model_key"] == model)
                    & (df["method"] == method)
                    & (df["metric"] == metric)
                ]

                if rows.empty:
                    continue

                row = rows.iloc[0]
                value = row["display_value"]
                if pd.notna(value):
                    value = np.clip(value, vmin, vmax)
                mat[i, j] = value
                plausible[i, j] = bool(row["plausible"])
                observed[i, j] = True

        last_im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        ax.set_title(METHOD_LABELS.get(method, method), fontsize=11)

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(
            [METRIC_LABELS.get(m, m) for m in metrics],
            rotation=45,
            ha="right",
            fontsize=9,
        )

        ax.set_yticks(np.arange(n_rows))
        if ax_idx == 0:
            ax.set_yticklabels(
                [MODEL_LABELS.get(model, model) for _, model in row_items],
                fontsize=8,
            )
        else:
            ax.tick_params(axis="y", labelleft=False)

        # Keep first row at the top
        ax.set_ylim(n_rows - 0.5, -0.5)

        # Cell gridlines
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        # --------------------------------------------------------
        # Markers
        # --------------------------------------------------------
        for i in range(n_rows):
            for j in range(n_cols):
                if not observed[i, j]:
                    ax.text(
                        j, i, "–",
                        ha="center", va="center",
                        fontsize=9, color="0.35"
                    )
                elif plausible[i, j]:
                    ax.scatter(
                        j, i,
                        s=28,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.1,
                        zorder=4,
                    )
                else:
                    ax.text(
                        j, i, "×",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black", zorder=4,
                    )

        # --------------------------------------------------------
        # Dataset separator lines
        # Extend into left margin on the first panel only.
        # --------------------------------------------------------
        sep_trans = blended_transform_factory(ax.transAxes, ax.transData)

        for block_idx, block in enumerate(dataset_blocks[:-1]):
            sep_y = block["end_row"] + 0.5

            if ax_idx == 0:
                x0 = -0.42   # extend into left margin
            else:
                x0 = 0.0

            ax.plot(
                [x0, 1.0],
                [sep_y, sep_y],
                transform=sep_trans,
                color="black",
                linewidth=1.1,
                alpha=0.9,
                clip_on=False,
                zorder=5,
            )

        # --------------------------------------------------------
        # Dataset labels in far-left margin of first panel only
        # --------------------------------------------------------
        if ax_idx == 0:
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            for block in dataset_blocks:
                ax.text(
                    -0.36,
                    block["center_row"],
                    block["dataset"],
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                    transform=trans,
                    clip_on=False,
                )

    fig.suptitle(
        "Monte Carlo plausibility scorecard",
        fontsize=13,
        y=0.985,
    )

    fig.text(
        0.5,
        0.025,
        r"○ = fail to reject / plausible; × = reject.",
        ha="center",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.22,
        right=0.90,
        top=0.91,
        bottom=0.17,
        wspace=0.08,
    )

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.19, 0.022, 0.67])

        cbar = fig.colorbar(
            last_im,
            cax=cbar_ax,
            extend="both",
        )

        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.ax.tick_params(labelsize=11)

        cbar.set_label(
            r"$(\mu_m - GT_m)/\sigma_m$",
            fontsize=13,
            labelpad=12,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Aggregate LaTeX table
# ---------------------------------------------------------------------

def make_aggregate_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    Create compact aggregate pass-rate table.

    Rows: realization methods.
    Columns: metrics plus overall.
    Entry: percent of tests that pass, where pass means fail to reject.
    """
    methods = ordered_present(df["method"], METHOD_ORDER)
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"])]

    # Compact labels for a one-column table.
    table_method_labels = {
        "EDGEIND": "EDGEIND",
        "LOCLBDG": "LOCLBDG",
        "PARABDG": "PARABDG",
    }

    lines: list[str] = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Aggregate Monte Carlo plausibility results. "
        r"Entries report the percentage of tests in which the ground-truth metric "
        r"was not rejected by the two-sided Monte Carlo test.}"
    )
    lines.append(r"\label{tab:aggregate-plausibility}")

    col_spec = "l" + "c" * (len(metrics) + 1)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = ["Method"] + [LATEX_METRIC_LABELS.get(m, m) for m in metrics] + ["Overall"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for method in methods:
        row_cells = [table_method_labels.get(method, method)]

        method_df = df[df["method"] == method]

        for metric in metrics:
            sub = method_df[method_df["metric"] == metric]
            n = len(sub)
            k = int(sub["plausible"].sum())

            if n == 0:
                cell = "--"
            else:
                pct = 100.0 * k / n
                cell = rf"{pct:.0f}\%"

            row_cells.append(cell)

        n_all = len(method_df)
        k_all = int(method_df["plausible"].sum())

        if n_all == 0:
            overall = "--"
        else:
            pct_all = 100.0 * k_all / n_all
            overall = rf"{pct_all:.0f}\%"

        row_cells.append(overall)

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Full numerical LaTeX table
# ---------------------------------------------------------------------

def make_full_numerical_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    Create a longtable with all numerical results.

    This is intended for the appendix.
    """
    sort_dataset = {v: i for i, v in enumerate(DATASET_ORDER)}
    sort_model = {
        "ER": 0,
        "CL": 1,
        "SB": 2,
        "SBM": 2,
        "KR": 3,
    }
    sort_method = {v: i for i, v in enumerate(METHOD_ORDER)}
    sort_metric = {v: i for i, v in enumerate(METRIC_ORDER)}

    df = df.copy()
    df["_dataset_order"] = df["dataset"].map(sort_dataset).fillna(999)
    df["_model_order"] = df["model_key"].map(sort_model).fillna(999)
    df["_method_order"] = df["method"].map(sort_method).fillna(999)
    df["_metric_order"] = df["metric"].map(sort_metric).fillna(999)

    df = df.sort_values(
        ["_dataset_order", "dataset", "_model_order", "_method_order", "_metric_order"]
    )

    lines: list[str] = []

    lines.append(r"\begin{longtable}{lllrrrrrrc}")
    lines.append(
        r"\caption{Full numerical Monte Carlo hypothesis-test results. "
        r"The empirical interval is the 2.5--97.5 percentile interval of the generated graphs.}"
        r"\label{tab:full-hypothesis-results}\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Model & Method & Metric & Ground truth & Mean $\pm$ SD & "
        r"Emp. low & Emp. high & $p$ & Reject? \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Model & Method & Metric & Ground truth & Mean $\pm$ SD & "
        r"Emp. low & Emp. high & $p$ & Reject? \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{10}{r}{Continued on next page} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfoot")

    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in df.iterrows():
        dataset = latex_escape(row["dataset"])
        model = latex_escape(MODEL_LABELS.get(row["model_key"], row["model_key"]))
        method = latex_escape(METHOD_LABELS.get(row["method"], row["method"]))
        metric = LATEX_METRIC_LABELS.get(row["metric"], latex_escape(row["metric"]))

        real = fmt_num(row["real"])
        mean_sd = rf"{fmt_num(row['mean_sim'])} $\pm$ {fmt_num(row['std_sim'])}"
        ci_low = fmt_num(row["ci_low"])
        ci_high = fmt_num(row["ci_high"])
        p_value = fmt_pvalue(row["p_value"])
        reject = "Yes" if bool(row["reject"]) else "No"

        lines.append(
            " & ".join(
                [
                    dataset,
                    model,
                    method,
                    metric,
                    real,
                    mean_sd,
                    ci_low,
                    ci_high,
                    p_value,
                    reject,
                ]
            )
            + r" \\"
        )

    lines.append(r"\end{longtable}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(INPUT_CSV)
    df = preprocess(df_raw)

    make_heatmap(df, HEATMAP_OUT)
    make_aggregate_table(df, AGG_TABLE_OUT)
    make_full_numerical_table(df, FULL_TABLE_OUT)

    print(f"Wrote {HEATMAP_OUT}")
    print(f"Wrote {AGG_TABLE_OUT}")
    print(f"Wrote {FULL_TABLE_OUT}")