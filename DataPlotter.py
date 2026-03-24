"""
DataPlotter.py  –  Multi-run comparison plotter for MAE252 MetaDrive telemetry
===============================================================================
Usage (from terminal):
    python DataPlotter.py                           # compare runs defined in COMPARE_RUNS below
    python DataPlotter.py logs/run_a logs/run_b     # compare arbitrary run folders

Usage (as a module):
    from DataPlotter import load_csv, plot_comparison
    runs = [
        ("Baseline Stanley",      "logs/20260309_235536_Base_SL/telemetry.csv"),
        ("Baseline PurePursuit",  "logs/20260309_235924_Base_PP/telemetry.csv"),
    ]
    plot_comparison(runs, columns=["speed_kmh", "w_efficiency", "w_comfort", "w_safety"])

Available columns (telemetry.csv):
    timestep, elapsed_s, x, y, heading_deg,
    speed_ms, speed_kmh, vx, vy, accel_ms2,
    steering, throttle_brake, total_dist_m, lane_changes,
    on_broken_line, crash_vehicle, crash_object, reward, cost, current_lane_id,
    maneuver, reason, block_ahead, dist_to_block_m,
    vlm_notes, w_efficiency, w_comfort, w_safety
"""

import csv
import sys
from pathlib import Path
from typing import List, Tuple, Optional

LOG_ROOT        = Path(__file__).parent / "logs"
COMPARISON_DIR  = Path(__file__).parent / "comparison_plots"

# ── visual style ──────────────────────────────────────────────────────────────
FONT_SIZE   = 16
TITLE_SIZE  = 18
LINE_WIDTH  = 2.8

# First entry = reference/baseline (black solid), rest = comparison styles
_STYLES = [
    {"color": "black",   "linestyle": "-",   "linewidth": LINE_WIDTH, "alpha": 1.0},
    {"color": "#e74c3c", "linestyle": "--",  "linewidth": LINE_WIDTH, "alpha": 0.9},
    {"color": "#2980b9", "linestyle": "-.",  "linewidth": LINE_WIDTH, "alpha": 0.9},
    {"color": "#27ae60", "linestyle": ":",   "linewidth": LINE_WIDTH, "alpha": 0.9},
    {"color": "#8e44ad", "linestyle": "--",  "linewidth": LINE_WIDTH, "alpha": 0.9},
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _find_csv(run_path) -> Path:
    """Accept a file path, or a folder containing telemetry.csv / planner.csv."""
    p = Path(run_path)
    if p.is_file():
        return p
    for name in ("telemetry.csv", "planner.csv"):
        cand = p / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No telemetry.csv or planner.csv found in {p}")


def load_csv(path) -> dict:
    """Load a CSV file into {column: [values...]}."""
    path = Path(path)
    print(f"[DataPlotter] Loading {path}")
    data: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, [])
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(val)
    return data


def latest_run_folder() -> Path:
    folders = sorted(
        (d for d in LOG_ROOT.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    if not folders:
        raise FileNotFoundError(f"No run folders in {LOG_ROOT}")
    return folders[-1]


# ── smoothing ─────────────────────────────────────────────────────────────────

def _smooth(vals: list, window: int) -> list:
    """Centred moving average, numeric values only."""
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window // 2)
        hi = min(len(vals), lo + window)
        chunk = [v for v in vals[lo:hi] if isinstance(v, (int, float))]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


# ── main multi-run plot ───────────────────────────────────────────────────────

def plot_comparison(
    runs: List[Tuple[str, str]],
    columns: List[str],
    x_col: str = "total_dist_m",
    title: str = "",
    save_path: str = None,
    smooth_window: int = 3,
):
    """
    Overlay multiple runs in stacked subplots.

    Parameters
    ----------
    runs          : list of (label, csv_path_or_folder); first = reference (black solid)
    columns       : column names to plot – one subplot each
    x_col         : x-axis column (default: 'total_dist_m')
    title         : figure title
    save_path     : save to file instead of displaying
    smooth_window : moving-average window (1 = no smoothing)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise ImportError("pip install matplotlib")

    # resolve paths and load
    datasets: List[Tuple[str, dict]] = []
    for label, path in runs:
        csv_path = _find_csv(path)
        datasets.append((label, load_csv(csv_path)))

    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(15, 4.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(title or "Run Comparison", fontsize=TITLE_SIZE, fontweight="bold", y=0.995)

    for ax_idx, (ax, col) in enumerate(zip(axes, columns)):
        any_plotted = False
        for run_idx, (label, data) in enumerate(datasets):
            if col not in data or x_col not in data:
                continue
            style = _STYLES[run_idx % len(_STYLES)]
            pairs = [
                (x, y) for x, y in zip(data[x_col], data[col])
                if isinstance(x, float) and isinstance(y, float)
            ]
            if not pairs:
                continue
            xs, ys = zip(*pairs)
            if smooth_window > 1:
                ys = _smooth(list(ys), smooth_window)
            ax.plot(xs, ys, label=label, **style)
            any_plotted = True

        if not any_plotted:
            ax.text(0.5, 0.5, f'"{col}" not found', transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=FONT_SIZE)

        # Draw a horizontal average line for each run
        for run_idx, (label, data) in enumerate(datasets):
            if col not in data or x_col not in data:
                continue
            vals = [v for v in data[col] if isinstance(v, (int, float))]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            style = _STYLES[run_idx % len(_STYLES)]
            ax.axhline(avg, color=style["color"], linestyle=":",
                       linewidth=1.6, alpha=0.75)
            xs_all = [x for x in data[x_col] if isinstance(x, (int, float))]
            x_label_pos = min(xs_all) * 1.02 if xs_all else 0
            # Alternate above/below so labels don't overlap
            va = "bottom" if run_idx % 2 == 0 else "top"
            ax.text(x_label_pos, avg, f"avg {avg:.2f} ",
                    color=style["color"], fontsize=FONT_SIZE - 3,
                    va=va, ha="left", fontweight="bold")

        ylabel = col.replace("_", " ")
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE, fontweight="bold")
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        if ax_idx == 0:
            ax.legend(fontsize=FONT_SIZE, framealpha=0.85, loc="best",
                      handlelength=2.5, borderpad=0.8)

    xlabel = x_col.replace("_", " ") + ("  (m)" if "dist" in x_col else "")
    axes[-1].set_xlabel(xlabel, fontsize=FONT_SIZE, fontweight="bold")
    axes[-1].tick_params(axis="x", labelsize=FONT_SIZE - 2)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", transparent=True)
        print(f"[DataPlotter] Saved -> {save_path}")
    else:
        plt.show()


# ── summary bar-chart comparison ─────────────────────────────────────────────

def plot_summary_comparison(run_folders, save_path: str = None):
    """
    2×2 bar-chart grid comparing runs on key summary.csv scalar metrics.

    Parameters
    ----------
    run_folders : list of folder paths, each containing summary.csv
    save_path   : output PNG path (default: comparison_plots/summary_comparison.png)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    METRICS = [
        ("avg_speed_kmh",       "Avg Speed (km/h)",      "higher = faster"),
        ("lane_changes",        "Lane Changes",           "instruction compliance"),
        ("crash_vehicle_steps", "Crash Steps (vehicle)",  "lower = safer"),
        ("total_steps",         "Total Steps",             "more = longer run"),
    ]

    summaries, labels = [], []
    for folder in run_folders:
        csv_path = Path(folder) / "summary.csv"
        if not csv_path.exists():
            print(f"[DataPlotter] WARNING: no summary.csv in {folder}, skipping")
            continue
        data = load_csv(csv_path)
        summaries.append({k: v[0] if v else 0 for k, v in data.items()})
        labels.append(Path(folder).name)

    if not summaries:
        print("[DataPlotter] No summary data found – skipping summary plot.")
        return

    colors = [_STYLES[j % len(_STYLES)]["color"] for j in range(len(labels))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, title, note) in enumerate(METRICS):
        ax = axes[i]
        vals = [float(s.get(metric, 0)) for s in summaries]

        bars = ax.bar(labels, vals, color=colors, width=0.5,
                      edgecolor="black", linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, fontweight="bold",
            )

        ax.set_title(title, fontsize=FONT_SIZE, fontweight="bold")
        ax.set_ylabel(title, fontsize=FONT_SIZE - 2)
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 3)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=FONT_SIZE - 5,
                color="gray", style="italic")

    fig.suptitle("Run Summary Comparison", fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()

    COMPARISON_DIR.mkdir(exist_ok=True)
    out = save_path or str(COMPARISON_DIR / "summary_comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", transparent=True)
    print(f"[DataPlotter] Summary plot saved -> {out}")
    plt.show()


# ── backwards-compatible single-run helpers ───────────────────────────────────

def load(csv_path=None) -> dict:
    if csv_path is None:
        csv_path = _find_csv(latest_run_folder())
    return load_csv(csv_path)


def plot_run(columns: list, csv_path=None, x_col: str = "total_dist_m",
             title: str = "", save_path: str = None):
    """Single-run convenience wrapper."""
    path = csv_path or _find_csv(latest_run_folder())
    label = Path(path).parent.name
    plot_comparison([(label, path)], columns,
                    x_col=x_col, title=title or label, save_path=save_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Runs to compare ───────────────────────────────────────────────────────
    # First entry = reference (black solid line).
    # Override from CLI: python DataPlotter.py logs/run_a logs/run_b
    # pull out --save flag if present
    args = sys.argv[1:]
    save_arg = None
    if "--save" in args:
        idx = args.index("--save")
        args.pop(idx)
        save_arg = args.pop(idx) if idx < len(args) and not args[idx].startswith("--") else None

    if args:
        COMPARE_RUNS = [(Path(p).name, p) for p in args]

    # Auto-save to comparison_plots/ unless explicitly disabled
    if save_arg is None:
        COMPARISON_DIR.mkdir(exist_ok=True)
        run_names = "_vs_".join(Path(r[1]).name for r in COMPARE_RUNS)
        save_arg = str(COMPARISON_DIR / f"{run_names}.png")
    else:
        COMPARE_RUNS = [
            ("Base Stanley",      "logs/20260309_235536_Base_SL"),
            ("Base PurePursuit",  "logs/20260309_235924_Base_PP"),
        ]

    # ── Columns to plot (one subplot each) ───────────────────────────────────
    COLUMNS = [
        "speed_kmh",
        "steering",
        "accel_ms2",
    ]
    # ─────────────────────────────────────────────────────────────────────────

    plot_comparison(
        COMPARE_RUNS,
        columns=COLUMNS,
        x_col="total_dist_m",
        title="Stanley vs Pure Pursuit – Baseline Comparison",
        smooth_window=5,
        save_path=save_arg,
    )

    plot_summary_comparison([r[1] for r in COMPARE_RUNS])
