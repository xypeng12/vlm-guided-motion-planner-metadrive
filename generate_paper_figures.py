"""
generate_paper_figures.py  â€“  Comparison plots + LaTeX tables for class paper
=============================================================================
Reads all 8 runs from logs/ and produces:

  comparison_plots/
    1_summary_bars.png              â€“ bar chart of key metrics (8 runs)
    2_timeline_profiles.png         â€“ stacked: speed, steering, crashes, lane changes
    3_controller_comparison.png     â€“ 2Ã—5 grid: base vs VLM, PP vs Stanley
    4_curve_comparison.png          â€“ 2Ã—5 grid: base vs VLM, B-Spline vs Clothoid
    5_vlm_effect_spider.png         â€“ radar chart: Baseline vs VLM averages
    paper_tables.tex                â€“ LaTeX tables + smoothness equation

Usage:
    python generate_paper_figures.py
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# â”€â”€ Global settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DPI        = 300
FONT_SIZE  = 13
TITLE_SIZE = 15
SAVEFIG_KW = dict(dpi=DPI, bbox_inches="tight", transparent=True, format="svg")

LOG_ROOT = Path(__file__).parent / "logs"
OUT_DIR  = Path(__file__).parent / "comparison_plots"
OUT_DIR.mkdir(exist_ok=True)

# All 8 runs: (short_label, folder_name, controller, curve, vlm)
RUNS = [
    ("PP+BS",          "Base_PPursuit_BS",       "Pure Pursuit", "B-Spline", False),
    ("PP+CL",          "Base_PPursuit_CL",       "Pure Pursuit", "Clothoid", False),
    ("ST+BS",          "Base_Stanley_BS",         "Stanley",      "B-Spline", False),
    ("ST+CL",          "Base_Stanley_CL",         "Stanley",      "Clothoid", False),
    ("VLM+PP+BS",      "VLM_safe_PPursuit_BS",   "Pure Pursuit", "B-Spline", True),
    ("VLM+PP+CL",      "VLM_safe_PPursuit_CL",   "Pure Pursuit", "Clothoid", True),
    ("VLM+ST+BS",      "VLM_safe_Stanley_BS",    "Stanley",      "B-Spline", True),
    ("VLM+ST+CL",      "VLM_safe_Stanley_CL",    "Stanley",      "Clothoid", True),
]

# â”€â”€ Color scheme â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Without VLM (solid): warm green-yellow family
# With VLM   (dashed): cool blue-purple family
COLOR_MAP = {
    "PP+BS":      "#091413",   # deep forest
    "PP+CL":      "#285A48",   # dark green
    "ST+BS":      "#408A71",   # teal green
    "ST+CL":      "#B0E4CC",   # mint
    "VLM+PP+BS":  "#0E21A0",   # deep blue
    "VLM+PP+CL":  "#4D2FB2",   # violet
    "VLM+ST+BS":  "#B153D7",   # orchid
    "VLM+ST+CL":  "#F375C2",   # pink
}
STYLE_MAP = {}  # populated after data load


# â”€â”€ I/O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_csv(path: Path) -> Dict[str, list]:
    data: Dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                k = k.strip()
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(v)
    return data


def load_summary(folder: Path) -> dict:
    path = folder / "summary.csv"
    if not path.exists():
        return {}
    d = load_csv(path)
    return {k: v[0] if v else None for k, v in d.items()}


def load_telemetry(folder: Path) -> Dict[str, list]:
    path = folder / "telemetry.csv"
    if not path.exists():
        return {}
    return load_csv(path)


def load_planner(folder: Path) -> Dict[str, list]:
    path = folder / "planner.csv"
    if not path.exists():
        return {}
    return load_csv(path)


def smooth(ys, w=7):
    if w <= 1 or len(ys) < w:
        return ys
    kernel = np.ones(w) / w
    return np.convolve(ys, kernel, mode="same").tolist()


# Will be populated after all runs are loaded to allow cross-run normalization
_raw_steer_rms: Dict[str, float] = {}
_raw_accel_rms: Dict[str, float] = {}

def _compute_raw_rms(telem: dict) -> Tuple[float, float]:
    """Return (RMS_delta_steering, RMS_delta_accel) for one run."""
    steer = [v for v in telem.get("steering", []) if isinstance(v, (int, float))]
    accel = [v for v in telem.get("accel_ms2", []) if isinstance(v, (int, float))]
    if len(steer) < 2 or len(accel) < 2:
        return (float("inf"), float("inf"))
    ds = np.diff(steer)
    da = np.diff(accel)
    return (float(np.sqrt(np.mean(ds ** 2))), float(np.sqrt(np.mean(da ** 2))))


def _normalize_smoothness():
    """
    Normalize each component (steering-jerk, accel-jerk) by cross-run mean,
    then combine 50/50.  This amplifies relative differences between runs.

    S = 0.5 * (RMS_steer / mean_RMS_steer) + 0.5 * (RMS_accel / mean_RMS_accel)
    """
    if not _raw_steer_rms:
        return
    mean_s = np.mean(list(_raw_steer_rms.values()))
    mean_a = np.mean(list(_raw_accel_rms.values()))
    if mean_s == 0:
        mean_s = 1.0
    if mean_a == 0:
        mean_a = 1.0
    for lb in _raw_steer_rms:
        smoothness_scores[lb] = (0.5 * _raw_steer_rms[lb] / mean_s
                                 + 0.5 * _raw_accel_rms[lb] / mean_a)


# â”€â”€ Load everything â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("[*] Loading data ...")
summaries: Dict[str, dict] = {}
telemetries: Dict[str, dict] = {}
planners: Dict[str, dict] = {}
smoothness_scores: Dict[str, float] = {}
valid_runs: List[Tuple[str, str, str, str, bool]] = []

for label, folder_name, ctrl, curve, vlm in RUNS:
    folder = LOG_ROOT / folder_name
    s = load_summary(folder)
    if not s:
        print(f"  WARN: {folder_name} has no summary -- skipping")
        continue
    t = load_telemetry(folder)
    summaries[label] = s
    telemetries[label] = t
    planners[label] = load_planner(folder)
    sr, ar = _compute_raw_rms(t)
    _raw_steer_rms[label] = sr
    _raw_accel_rms[label] = ar
    valid_runs.append((label, folder_name, ctrl, curve, vlm))

# Normalize smoothness scores across all runs
_normalize_smoothness()

labels = [r[0] for r in valid_runs]
base_labels = [lb for lb, _, _, _, v in valid_runs if not v]
vlm_labels  = [lb for lb, _, _, _, v in valid_runs if v]

# Build style map: base = solid, VLM = dashed
for lb, _, _, _, v in valid_runs:
    STYLE_MAP[lb] = {
        "color":     COLOR_MAP.get(lb, "#333333"),
        "linestyle": "--" if v else "-",
        "linewidth": 2.0,
        "alpha":     0.85,
    }

print(f"  Loaded {len(valid_runs)} runs: {labels}")
print(f"  Smoothness scores: { {lb: round(v, 4) for lb, v in smoothness_scores.items()} }")


# â”€â”€ Helper: average a metric (from summary or smoothness) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _avg_metric(filter_fn, metric):
    if metric == "smoothness":
        vals = [smoothness_scores[lb]
                for lb, _, ctrl, curve, vlm in valid_runs if filter_fn(ctrl, curve, vlm)]
    else:
        vals = [float(summaries[lb].get(metric, 0))
                for lb, _, ctrl, curve, vlm in valid_runs if filter_fn(ctrl, curve, vlm)]
    return sum(vals) / len(vals) if vals else 0


# =====================================================================
# PLOT 1: Summary bar chart (2x3 grid)
# =====================================================================

def plot_summary_bars():
    METRICS = [
        ("total_steps",         "Total Steps",            "lower = faster completion"),
        ("avg_speed_kmh",       "Avg Speed (km/h)",       "higher = faster"),
        ("lane_changes",        "Lane Changes",           "fewer = more conservative"),
        ("crash_vehicle_steps", "Crash Steps",            "lower = safer"),
        ("smoothness",          "Smoothness Score",       "lower = smoother"),
        ("broken_line_steps",   "Broken Line Steps",      "lower = smoother"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (metric, title, note) in enumerate(METRICS):
        ax = axes[i]
        if metric == "smoothness":
            vals = [round(smoothness_scores.get(lb, 0), 3) for lb in labels]
        else:
            vals = [float(summaries[lb].get(metric, 0)) for lb in labels]
        colors = [COLOR_MAP.get(lb, "#888") for lb in labels]

        bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6,
                      edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.3f}" if metric == "smoothness" else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom",
                    fontsize=FONT_SIZE - 3, fontweight="bold")

        ax.set_title(title, fontsize=FONT_SIZE, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([""] * len(labels))  # hide per-subplot labels
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=FONT_SIZE - 5,
                color="gray", style="italic")

    # Shared legend at the bottom
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=COLOR_MAP.get(lb, "#888"),
                            edgecolor="black", linewidth=0.8, label=lb)
                      for lb in labels]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(labels), fontsize=FONT_SIZE - 2,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Performance Summary -- All Configurations",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "Res_fig_1_summary_bars.svg"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close()
    print(f"  -> {out}")


# =====================================================================
# PLOT 2: Timeline profiles (4 vertical subplots, single legend)
# =====================================================================

def plot_timeline_profiles():
    PANELS = [
        ("speed_kmh",     "Speed (km/h)",             "smooth",  11),
        ("steering",      "Steering Angle",           "smooth",  15),
        ("crash_vehicle", "Cumulative Crash Steps",   "cumsum",  1),
        ("lane_changes",  "Cumulative Lane Changes",  "raw",     1),
    ]

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(14, 4.0 * len(PANELS)),
                             sharex=True)

    for ax, (col, ylabel, mode, sw) in zip(axes, PANELS):
        for lb in labels:
            t = telemetries.get(lb, {})
            xs = t.get("total_dist_m", [])
            ys = t.get(col, [])
            if not xs or not ys:
                continue
            pairs = [(x, y) for x, y in zip(xs, ys)
                     if isinstance(x, (int, float)) and isinstance(y, (int, float))]
            if not pairs:
                continue
            xs2, ys2 = zip(*pairs)
            if mode == "cumsum":
                ys2 = np.cumsum(ys2).tolist()
            elif mode == "smooth":
                ys2 = smooth(list(ys2), sw)
            ax.plot(xs2, ys2, label=lb, **STYLE_MAP[lb])

        ax.set_ylabel(ylabel, fontsize=FONT_SIZE, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.tick_params(labelsize=FONT_SIZE - 2)

    # Single legend at bottom
    axes[0].set_title("Driving Profiles Over Distance", fontsize=TITLE_SIZE,
                      fontweight="bold")
    axes[-1].set_xlabel("Distance (m)", fontsize=FONT_SIZE, fontweight="bold")
    axes[-1].legend(fontsize=FONT_SIZE - 3, ncol=4, loc="upper center",
                    bbox_to_anchor=(0.5, -0.25), framealpha=0.9)

    plt.tight_layout()
    out = OUT_DIR / "Res_fig_2_timeline_profiles.svg"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close()
    print(f"  -> {out}")


# =====================================================================
# PLOT 3 & 4: Controller / Curve comparison -- 2x5 grids
# =====================================================================

def _comparison_2xN(group_key, group_values, group_colors, title, filename):
    """
    2xN grid.  Left column = without VLM, right = with VLM.
    Rows: Avg Speed, Total Steps, Smoothness, Lane Changes, Crash Steps.
    """
    METRICS = [
        ("avg_speed_kmh",       "Avg Speed (km/h)"),
        ("total_steps",         "Total Steps"),
        ("smoothness",          "Smoothness Score"),
        ("lane_changes",        "Lane Changes"),
        ("crash_vehicle_steps", "Crash Steps"),
    ]
    n_rows = len(METRICS)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.6 * n_rows))

    for col_idx, (vlm_flag, col_title) in enumerate(
            [(False, "Without VLM"), (True, "With VLM")]):
        for row_idx, (mkey, mname) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            vals, bar_labels = [], []
            for gv in group_values:
                if mkey == "smoothness":
                    vs = [smoothness_scores[lb]
                          for lb, _, ctrl, curve, vlm in valid_runs
                          if ((ctrl if group_key == "controller" else curve) == gv
                              and vlm == vlm_flag)]
                else:
                    vs = [float(summaries[lb].get(mkey, 0))
                          for lb, _, ctrl, curve, vlm in valid_runs
                          if ((ctrl if group_key == "controller" else curve) == gv
                              and vlm == vlm_flag)]
                avg = sum(vs) / len(vs) if vs else 0
                vals.append(avg)
                bar_labels.append(gv)

            y_pos = np.arange(len(bar_labels))
            bars = ax.barh(y_pos, vals,
                           color=group_colors[:len(bar_labels)],
                           height=0.30, edgecolor="none")
            for bar, val in zip(bars, vals):
                fmt = f"{val:.3f}" if mkey == "smoothness" else f"{val:.1f}"
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                        f"  {fmt}", va="center",
                        fontsize=FONT_SIZE - 2, fontweight="bold")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(bar_labels, fontsize=FONT_SIZE - 2,
                               rotation=90, va="center")
            ax.grid(axis="x", alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=FONT_SIZE - 2)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=FONT_SIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(mname, fontsize=FONT_SIZE - 1, fontweight="bold",
                              rotation=90, labelpad=10)

    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, **SAVEFIG_KW)
    plt.close()
    print(f"  -> {out}")


def plot_controller_comparison():
    _comparison_2xN(
        group_key="controller",
        group_values=["Pure Pursuit", "Stanley"],
        group_colors=["#285A48", "#408A71"],
        title="Controller Comparison (Pure Pursuit vs Stanley)",
        filename="Res_fig_3_controller_comparison.svg",
    )


def plot_curve_comparison():
    _comparison_2xN(
        group_key="curve",
        group_values=["B-Spline", "Clothoid"],
        group_colors=["#4D2FB2", "#B153D7"],
        title="Curve Generator Comparison (B-Spline vs Clothoid)",
        filename="Res_fig_4_curve_comparison.svg",
    )


# =====================================================================
# PLOT 5: VLM effect -- spider / radar chart
# =====================================================================

def plot_vlm_spider():
    METRICS = [
        ("avg_speed_kmh",       "Avg Speed",     True),   # higher = better
        ("total_steps",         "Total Steps",   False),  # lower = better
        ("smoothness",          "Smoothness",    False),  # lower = better
        ("lane_changes",        "Lane Changes",  False),  # lower = better
        ("crash_vehicle_steps", "Crash Steps",   False),  # lower = better
    ]

    base_vals, vlm_vals = [], []
    for mkey, _, _ in METRICS:
        base_vals.append(_avg_metric(lambda c, cu, v: not v, mkey))
        vlm_vals.append(_avg_metric(lambda c, cu, v: v,     mkey))

    # Normalize to [0.15, 1.0] where 1 = best  (floor avoids collapse to center)
    FLOOR = 0.15
    norm_base, norm_vlm = [], []
    for i, (_, _, higher_better) in enumerate(METRICS):
        all_v = [base_vals[i], vlm_vals[i]]
        lo, hi = min(all_v), max(all_v)
        rng = hi - lo if hi != lo else 1.0
        if higher_better:
            nb = (base_vals[i] - lo) / rng
            nv = (vlm_vals[i] - lo) / rng
        else:
            nb = 1.0 - (base_vals[i] - lo) / rng
            nv = 1.0 - (vlm_vals[i] - lo) / rng
        # Rescale [0,1] -> [FLOOR, 1.0]
        norm_base.append(FLOOR + nb * (1.0 - FLOOR))
        norm_vlm.append(FLOOR + nv * (1.0 - FLOOR))

    cat_labels = [m[1] for m in METRICS]
    N = len(cat_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    norm_base += norm_base[:1]
    norm_vlm  += norm_vlm[:1]

    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_axes([0.1, 0.25, 0.8, 0.7], polar=True)

    # Remove radial spoke lines, keep concentric grid
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], [])
    ax.spines["polar"].set_visible(False)

    # Baseline polygon (green fill)
    ax.fill(angles, norm_base, color="#408A71", alpha=0.25)
    ax.plot(angles, norm_base, color="#285A48", linewidth=2.5,
            marker="o", markersize=6, label="Baseline")

    # VLM polygon (cool purple fill)
    ax.fill(angles, norm_vlm, color="#B153D7", alpha=0.25)
    ax.plot(angles, norm_vlm, color="#4D2FB2", linewidth=2.5,
            marker="s", markersize=6, label="VLM (safe)")

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=FONT_SIZE, fontweight="bold")

    # Concentric grid rings
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])  # hide numeric rings
    ax.set_ylim(0, 1.10)
    ax.yaxis.grid(True, alpha=0.3, linestyle="-", color="gray")

    ax.legend(fontsize=FONT_SIZE, loc="upper right",
              bbox_to_anchor=(1.25, 1.08))

    # Raw-value table below the radar chart
    col_labels = [m[1] for m in METRICS]
    b_cells = [f"{v:.3f}" if mk == "smoothness" else f"{v:.1f}"
               for (mk, _, _), v in zip(METRICS, base_vals)]
    v_cells = [f"{v:.3f}" if mk == "smoothness" else f"{v:.1f}"
               for (mk, _, _), v in zip(METRICS, vlm_vals)]
    table_ax = fig.add_axes([0.08, 0.02, 0.84, 0.14])
    table_ax.axis("off")
    tbl = table_ax.table(
        cellText=[b_cells, v_cells],
        rowLabels=["Baseline", "VLM (safe)"],
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZE - 2)
    tbl.scale(1.0, 1.5)
    # Color the row labels
    for (row, col), cell in tbl.get_celld().items():
        if col == -1 and row == 1:
            cell.set_text_props(color="#285A48", fontweight="bold")
        elif col == -1 and row == 2:
            cell.set_text_props(color="#4D2FB2", fontweight="bold")
        if row == 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#cccccc")
    out = OUT_DIR / "Res_fig_5_vlm_effect_spider.svg"
    fig.savefig(out, **SAVEFIG_KW)
    plt.close()
    print(f"  -> {out}")


# =====================================================================
# LaTeX Tables
# =====================================================================

def generate_latex():
    lines = []

    # -- Smoothness equation ---------------------------------------------------
    lines.append(r"% ============================================================")
    lines.append(r"% Smoothness Score Definition")
    lines.append(r"% ============================================================")
    lines.append(r"% The smoothness metric captures ride jerkiness by combining")
    lines.append(r"% steering rate and longitudinal jerk.")
    lines.append(r"% Lower values indicate smoother driving.")
    lines.append(r"%")
    lines.append(r"% Equation (paste into your paper):")
    lines.append(r"\begin{equation}")
    lines.append(r"  S_{\text{smooth}} = \frac{1}{2}\,"
                 r"\frac{\mathrm{RMS}(\Delta\delta)}"
                 r"{\overline{\mathrm{RMS}(\Delta\delta)}}")
    lines.append(r"  \;+\; \frac{1}{2}\,"
                 r"\frac{\mathrm{RMS}(\Delta a)}"
                 r"{\overline{\mathrm{RMS}(\Delta a)}}")
    lines.append(r"  \label{eq:smoothness}")
    lines.append(r"\end{equation}")
    lines.append(r"% where RMS(\Delta\delta) = sqrt(mean((\delta_{i+1}-\delta_i)^2)),")
    lines.append(r"% and the overline denotes the cross-run mean of that component.")
    lines.append(r"% Normalizing by the cross-run mean ensures both components")
    lines.append(r"% contribute equally regardless of their raw magnitude.")
    lines.append("")

    # -- Table 1: Full comparison ----------------------------------------------
    lines.append(r"% ============================================================")
    lines.append(r"% Table 1: Full performance comparison")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance comparison across all configurations. "
                 r"Bold = best. Smoothness defined in Eq.~\eqref{eq:smoothness}.}")
    lines.append(r"\label{tab:full_comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Steps & Avg Speed & Lane & Crash "
                 r"& Smoothness & Broken \\")
    lines.append(r"              &       & (km/h)    & Changes & Steps "
                 r"& Score & Lines \\")
    lines.append(r"\midrule")

    metric_keys = ["total_steps", "avg_speed_kmh", "lane_changes",
                   "crash_vehicle_steps", "smoothness", "broken_line_steps"]

    def _get_val(lb, mk):
        if mk == "smoothness":
            return smoothness_scores.get(lb, 999)
        return float(summaries[lb].get(mk, 9999))

    best = {}
    for mk in metric_keys:
        vals = [_get_val(lb, mk) for lb in labels]
        best[mk] = max(vals) if mk == "avg_speed_kmh" else min(vals)

    for idx, lb in enumerate(labels):
        cells = []
        for mk in metric_keys:
            v = _get_val(lb, mk)
            if mk == "avg_speed_kmh":
                fmt = f"{v:.1f}"
            elif mk == "smoothness":
                fmt = f"{v:.4f}"
            else:
                fmt = f"{v:.0f}"
            if abs(v - best[mk]) < 1e-6:
                cells.append(r"\textbf{" + fmt + "}")
            else:
                cells.append(fmt)
        if idx == 4:
            lines.append(r"\midrule")
        lines.append(f"{lb} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # -- Table 2: Controller comparison ----------------------------------------
    lines.append(r"% ============================================================")
    lines.append(r"% Table 2: Controller comparison (averaged)")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Controller comparison averaged over curve generator "
                 r"and VLM settings.}")
    lines.append(r"\label{tab:controller_comparison}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Controller & Avg Speed & Crash Steps & Smoothness "
                 r"& Lane Changes & Broken Lines \\")
    lines.append(r"\midrule")
    for ctrl in ["Pure Pursuit", "Stanley"]:
        vals = [_avg_metric(lambda c, cu, v, _c=ctrl: c == _c, mk)
                for mk in ["avg_speed_kmh", "crash_vehicle_steps", "smoothness",
                            "lane_changes", "broken_line_steps"]]
        lines.append(f"{ctrl} & {vals[0]:.1f} & {vals[1]:.1f} & "
                     f"{vals[2]:.4f} & {vals[3]:.1f} & {vals[4]:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # -- Table 3: Curve generator comparison -----------------------------------
    lines.append(r"% ============================================================")
    lines.append(r"% Table 3: Curve generator comparison (averaged)")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Curve generator comparison averaged over controller "
                 r"and VLM settings.}")
    lines.append(r"\label{tab:curve_comparison}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Curve Generator & Avg Speed & Crash Steps & Smoothness "
                 r"& Lane Changes & Broken Lines \\")
    lines.append(r"\midrule")
    for cu in ["B-Spline", "Clothoid"]:
        vals = [_avg_metric(lambda c, cu2, v, _cu=cu: cu2 == _cu, mk)
                for mk in ["avg_speed_kmh", "crash_vehicle_steps", "smoothness",
                            "lane_changes", "broken_line_steps"]]
        lines.append(f"{cu} & {vals[0]:.1f} & {vals[1]:.1f} & "
                     f"{vals[2]:.4f} & {vals[3]:.1f} & {vals[4]:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # -- Table 4: VLM effect ---------------------------------------------------
    lines.append(r"% ============================================================")
    lines.append(r"% Table 4: VLM effect (averaged)")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of VLM-guided planning with safe instruction, "
                 r"averaged over all controller and curve combinations.}")
    lines.append(r"\label{tab:vlm_effect}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Mode & Avg Speed & Total Steps & Smoothness "
                 r"& Lane Changes & Crash Steps \\")
    lines.append(r"\midrule")
    for mode, filt in [("Baseline", lambda c, cu, v: not v),
                       ("VLM (safe)", lambda c, cu, v: v)]:
        vals = [_avg_metric(filt, mk) for mk in
                ["avg_speed_kmh", "total_steps", "smoothness",
                 "lane_changes", "crash_vehicle_steps"]]
        lines.append(f"{mode} & {vals[0]:.1f} & {vals[1]:.0f} & "
                     f"{vals[2]:.4f} & {vals[3]:.1f} & {vals[4]:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # -- Table 5: Pairwise VLM improvement -------------------------------------
    lines.append(r"% ============================================================")
    lines.append(r"% Table 5: Pairwise VLM improvement per configuration")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pairwise change from adding VLM (safe). "
                 r"Negative $\Delta$ = improvement for cost/crash/smoothness.}")
    lines.append(r"\label{tab:pairwise_vlm}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Config Pair & $\Delta$ Steps & $\Delta$ Speed "
                 r"& $\Delta$ Smoothness & $\Delta$ Crashes & $\Delta$ Lane Ch. \\")
    lines.append(r"\midrule")

    pairs = [("PP+BS", "VLM+PP+BS"), ("PP+CL", "VLM+PP+CL"),
             ("ST+BS", "VLM+ST+BS"), ("ST+CL", "VLM+ST+CL")]
    for base_lb, vlm_lb in pairs:
        if base_lb not in summaries or vlm_lb not in summaries:
            continue
        sb, sv = summaries[base_lb], summaries[vlm_lb]
        d_steps  = float(sv.get("total_steps", 0)) - float(sb.get("total_steps", 0))
        d_speed  = (float(sv.get("avg_speed_kmh", 0))
                    - float(sb.get("avg_speed_kmh", 0)))
        d_smooth = (smoothness_scores.get(vlm_lb, 0)
                    - smoothness_scores.get(base_lb, 0))
        d_crash  = (float(sv.get("crash_vehicle_steps", 0))
                    - float(sb.get("crash_vehicle_steps", 0)))
        d_lc     = (float(sv.get("lane_changes", 0))
                    - float(sb.get("lane_changes", 0)))

        def _fd(v, fmt=".0f"):
            return f"{'+' if v > 0 else ''}{v:{fmt}}"

        lines.append(
            f"{base_lb} $\\to$ {vlm_lb} & {_fd(d_steps)} & "
            f"{_fd(d_speed, '.1f')} & {_fd(d_smooth, '.4f')} & "
            f"{_fd(d_crash)} & {_fd(d_lc)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    out = OUT_DIR / "paper_tables.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"  -> {out}")
    return tex_content


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("\n[1/5] Summary bar chart ...")
    plot_summary_bars()

    print("[2/5] Timeline profiles (stacked) ...")
    plot_timeline_profiles()

    print("[3/5] Controller comparison (2x5) ...")
    plot_controller_comparison()

    print("[4/5] Curve generator comparison (2x5) ...")
    plot_curve_comparison()

    print("[5/5] VLM effect spider chart ...")
    plot_vlm_spider()

    print("\n[LaTeX] Generating tables ...")
    tex = generate_latex()

    print("\n" + "=" * 60)
    print("All plots saved to: comparison_plots/")
    print("LaTeX tables saved to: comparison_plots/paper_tables.tex")
    print("=" * 60)
    print("\nLaTeX output preview:\n")
    print(tex)
