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

DATASET_SHORT = {
    "hamsterster": "Hams",
    "facebook": "Fcbk",
    "polblogs": "Polb",
    "web-spam": "Spam",
    "bio-CE-PG": "Cepg",
    "bio-SC-HT": "Scht",
}

METHOD_SHORT = {
    "EDGEIND": "EDGE",
    "LOCLBDG": "LOCL",
    "PARABDG": "PARA",
}

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

def fmt_pvalue_compact(p: float) -> str:
    """Compact p-value formatter for dense appendix tables."""
    if pd.isna(p):
        return "--"

    p = float(p)

    if p < 0.001:
        return r"$<.001$"

    s = f"{p:.3f}"
    if s.startswith("0."):
        s = s[1:]
    return s


def fmt_interval_compact(low: float, high: float, metric: str) -> str:
    """
    Compact two-line empirical interval formatter.

    Produces:
      [low-
       high]

    k4_density is scaled consistently with the table header.
    """
    if pd.isna(low) or pd.isna(high):
        return "--"

    low_s = fmt_metric_num(low, metric)
    high_s = fmt_metric_num(high, metric)

    return rf"\shortstack{{[{low_s}-\\{high_s}]}}"


def metric_scale(metric: str) -> float:
    """
    Scale metrics for compact table display.

    # k4_density is multiplied by 1e4 and labeled as x10^4 in the header.
    """
    if metric == "k4_density":
        return 1e4
    return 1.0


def fmt_metric_num(x: float, metric: str, digits: int = 2) -> str:
    """
    Format a metric value for the dense appendix table.

    All values are rounded to two decimals, following the compact style
    of the original Table II. k4_density is displayed after multiplying
    by 1e4.
    """
    if pd.isna(x):
        return "--"

    value = float(x) * metric_scale(metric)
    return f"{value:.{digits}f}"


def fmt_pvalue_table(p: float) -> str:
    """Two-decimal p-value formatter for the dense appendix table."""
    if pd.isna(p):
        return "--"

    p = float(p)

    if p < 0.005 and p > 0:
        return r"$<0.01$"

    return f"{p:.2f}"


def fmt_decision(reject: bool) -> str:
    """
    Decision marker:
      x = reject
      o = fail to reject / plausible
    """
    return r"$\times$" if bool(reject) else r"$\circ$"


def model_variants(model_key: str) -> list[str]:
    """Allow SB/SBM aliases in the CSV."""
    if model_key in {"SB", "SBM"}:
        return ["SB", "SBM"]
    return [model_key]


def metric_label_header(metric: str) -> str:
    """
    Compact metric label for the dense table header.
    """
    labels = {
        "gcc": "GCC",
        "alcc": "ALCC",
        "k4_density": r"\shortstack{$K_4$ dens.\\$\times 10^4$}",
        "C3_global": r"$C_3$",
        "C3_avg_local": r"\shortstack{$C_3$\\local}",
    }
    return labels.get(metric, LATEX_METRIC_LABELS.get(metric, latex_escape(metric)))

def chunked(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    return [items[i:i + size] for i in range(0, len(items), size)]

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
    Create one full numerical appendix table in portrait orientation.

    The table is split into dataset panels of three datasets each, but it
    remains a single generated LaTeX file. Each panel follows a Table-II-like
    structure:

      - dataset groups across columns,
      - metric subcolumns within each dataset,
      - one GroundTruth row,
      - model/method blocks in the body.

    For each model/method/metric/dataset combination, four stacked rows are used:

      μ     generated-graph mean
      I95   empirical interval formatted as [low-high] over two lines
      p     two-sided Monte Carlo p-value
      D     decision marker: x = reject, o = fail to reject

    k4_density values are multiplied by 1e4 and labeled as (E-4).
    """
    datasets = ordered_present(df["dataset"], DATASET_ORDER)
    dataset_panels = chunked(datasets, 3)

    metrics = [m for m in METRIC_ORDER if m in set(df["metric"])]
    methods = ordered_present(df["method"], METHOD_ORDER)

    # Stable model order while accepting either SB or SBM from the CSV.
    raw_models = ordered_present(df["model_key"], MODEL_ORDER)

    models = []
    seen_model_family = set()
    for m in raw_models:
        family = "SB" if m in {"SB", "SBM"} else m
        if family not in seen_model_family:
            models.append(m)
            seen_model_family.add(family)

    method_labels = METHOD_SHORT

    # Portrait-oriented dense table:
    # 3 left columns + 3 datasets * 4 metrics = 15 columns.
    # Wider columns so the table fills the page better.
    model_col_width = 0.045
    method_col_width = 0.065
    entry_col_width = 0.045

    # Total target table width. Keep slightly under 1.0 to account for rules and padding.
    target_table_width = 0.965

    centered_p = r">{\centering\arraybackslash}p"

    def make_col_spec(panel_datasets: list[str]) -> str:
        n_metric_cols = len(panel_datasets) * len(metrics)

        fixed_width = model_col_width + method_col_width + entry_col_width
        metric_col_width = (target_table_width - fixed_width) / n_metric_cols

        col_spec = (
            rf"|{centered_p}{{{model_col_width:.4f}\linewidth}}"
            rf"|{centered_p}{{{method_col_width:.4f}\linewidth}}"
            rf"|{centered_p}{{{entry_col_width:.4f}\linewidth}}||"
        )

        for _dataset in panel_datasets:
            for _metric in metrics:
                col_spec += rf"{centered_p}{{{metric_col_width:.4f}\linewidth}}|"
            col_spec += "|"

        return col_spec

    def total_cols(panel_datasets: list[str]) -> int:
        return 3 + len(panel_datasets) * len(metrics)

    def append_panel_header(lines: list[str], panel_datasets: list[str]) -> None:
        # Dataset group row.
        row = [r"\multicolumn{3}{|c||}{dataset}"]
        for dataset in panel_datasets:
            dshort = latex_escape(DATASET_SHORT.get(dataset, dataset))
            row.append(rf"\multicolumn{{{len(metrics)}}}{{c||}}{{{dshort}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(row) + r" \\")

        # Metric subheader row.
        row = [r"\multicolumn{3}{|c||}{metric}"]
        for _dataset in panel_datasets:
            for metric in metrics:
                row.append(metric_label_header(metric))
        lines.append(r"\hline")
        lines.append(" & ".join(row) + r" \\")

        # Ground-truth row.
        row = [r"\multicolumn{3}{|c||}{GroundTruth}"]
        for dataset in panel_datasets:
            for metric in metrics:
                gt_rows = df[
                    (df["dataset"] == dataset)
                    & (df["metric"] == metric)
                    & (df["real"].notna())
                ]

                if gt_rows.empty:
                    row.append("--")
                else:
                    gt = float(gt_rows.iloc[0]["real"])
                    row.append(fmt_metric_num(gt, metric))

        lines.append(r"\hline")
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline\hline")

        # Body-side labels.
        row = ["Model", "Method", "Entry"]
        for _dataset in panel_datasets:
            for _metric in metrics:
                row.append("")
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")

    def append_panel_body(lines: list[str], panel_datasets: list[str]) -> None:
        # Four logical rows per method. I95 is one row with a two-line cell.
        entry_rows = [
            ("mean", r"$\mu$"),
            ("interval", r"$I_{95}$"),
            ("p", r"$p$"),
            ("decision", r"$D$"),
        ]

        for model in models:
            model_keys = model_variants(model)
            model_label = latex_escape(MODEL_LABELS.get(model, model))
            model_span = len(methods) * len(entry_rows)

            for method_idx, method in enumerate(methods):
                method_label = latex_escape(method_labels.get(method, method))
                method_span = len(entry_rows)

                for entry_idx, (entry_kind, entry_label) in enumerate(entry_rows):
                    row_cells: list[str] = []

                    # Model multirow only once per model block.
                    if method_idx == 0 and entry_idx == 0:
                        row_cells.append(
                            rf"\multirow{{{model_span}}}{{*}}{{\centering {model_label}}}"
                        )
                    else:
                        row_cells.append("")

                    # Method multirow once per method block.
                    if entry_idx == 0:
                        row_cells.append(
                            rf"\multirow{{{method_span}}}{{*}}{{\centering {method_label}}}"
                        )
                    else:
                        row_cells.append("")

                    row_cells.append(entry_label)

                    for dataset in panel_datasets:
                        for metric in metrics:
                            rows = df[
                                (df["dataset"] == dataset)
                                & (df["model_key"].isin(model_keys))
                                & (df["method"] == method)
                                & (df["metric"] == metric)
                            ]

                            if rows.empty:
                                row_cells.append("--")
                                continue

                            result = rows.iloc[0]

                            if entry_kind == "mean":
                                value = fmt_metric_num(result["mean_sim"], metric)
                            elif entry_kind == "interval":
                                value = fmt_interval_compact(
                                    result["ci_low"],
                                    result["ci_high"],
                                    metric,
                                )
                            elif entry_kind == "p":
                                value = fmt_pvalue_table(result["p_value"])
                            elif entry_kind == "decision":
                                value = fmt_decision(result["reject"])
                            else:
                                value = "--"

                            row_cells.append(value)

                    lines.append(" & ".join(row_cells) + r" \\")

                # Separator after each method block.
                lines.append(r"\hline")


    lines: list[str] = []

    lines.append(r"\begingroup")
    lines.append(r"\setlength{\LTcapwidth}{\textwidth}")
    lines.append(r"\setlength{\LTleft}{0pt}")
    lines.append(r"\setlength{\LTright}{0pt}")
    lines.append(r"\setlength{\tabcolsep}{1.6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\scriptsize")

    for panel_idx, panel_datasets in enumerate(dataset_panels):
        col_spec = make_col_spec(panel_datasets)
        ncols = total_cols(panel_datasets)

        lines.append(rf"\begin{{longtable}}{{{col_spec}}}")

        if panel_idx == 0:
            # Keep caption readable. Do not put it inside \tiny.
            lines.append(
                r"\caption{\small Full numerical Monte Carlo hypothesis-test results. "
                r"Columns are grouped by dataset and metric. The GroundTruth row reports the "
                r"observed metric value for each dataset. For each fitted model and realization "
                r"method, the stacked rows report $\mu$ (generated-graph mean), "
                r"$I_{95}$ (empirical 2.5--97.5 percentile interval), $p$ "
                r"(two-sided Monte Carlo p-value), and $D$ (decision), where "
                r"$\times$ indicates rejection and $\circ$ indicates failure to reject. "
                r"$K_4$-density entries are scaled by $10^4$ for readability. "
                r"EDGE, LOCL, and PARA denote edge-independent, local binding, and parallel "
                r"binding realization, respectively.}"
                r"\label{tab:full-hypothesis-results}\\"
            )
        else:
            lines.append(
                rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{Table~\ref{{tab:full-hypothesis-results}} continued}}}} \\"
            )

        append_panel_header(lines, panel_datasets)
        append_panel_body(lines, panel_datasets)

        lines.append(r"\end{longtable}")
        lines.append("")

        if panel_idx < len(dataset_panels) - 1:
            lines.append(r"\vspace{0.75em}")
            lines.append("")

    lines.append(r"\endgroup")
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