"""
DataPlotter.py  –  Simple time-series plotting for MAE252 MetaDrive telemetry
==============================================================================
Usage (from terminal):
    python DataPlotter.py                           # auto-picks latest run
    python DataPlotter.py logs/20240226_120000_highway/telemetry.csv

Usage (as a module):
    from DataPlotter import load, plot_columns
    df = load()                                     # latest run
    plot_columns(df, ["speed_kmh", "accel_ms2", "steering", "throttle_brake"])
"""

import csv
import os
import sys
from pathlib import Path

# ── available column names in telemetry.csv ───────────────────────────────────
# timestep, elapsed_s, x, y, heading_deg,
# speed_ms, speed_kmh, vx, vy, accel_ms2,
# steering, throttle_brake, total_dist_m, lane_changes,
# on_broken_line, crash_vehicle, crash_object, reward, cost, current_lane_id

LOG_ROOT = Path(__file__).parent / "logs"


# ── helpers ───────────────────────────────────────────────────────────────────

def latest_csv() -> Path:
    """Return the telemetry.csv from the most-recently-modified run folder."""
    folders = sorted(
        (d for d in LOG_ROOT.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    if not folders:
        raise FileNotFoundError(f"No run folders found in {LOG_ROOT}")
    csv_path = folders[-1] / "telemetry.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No telemetry.csv in {folders[-1]}")
    return csv_path


def load(csv_path=None) -> dict:
    """
    Load a telemetry CSV into a plain dict of lists.
    If csv_path is None, the latest run is used automatically.

    Returns
    -------
    dict  {column_name: [float, ...]}
    """
    if csv_path is None:
        csv_path = latest_csv()
    csv_path = Path(csv_path)
    print(f"[DataPlotter] Loading {csv_path}")

    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, [])
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(val)   # keep strings as-is
    return data


# ── main plotting function ────────────────────────────────────────────────────

def plot_columns(data: dict, columns: list, x_col: str = "elapsed_s",
                 title: str = "", save_path: str = None):
    """
    Plot selected columns as stacked subplots sharing the same x-axis.

    Parameters
    ----------
    data      : dict returned by load()
    columns   : list of column names to plot, one subplot each
    x_col     : x-axis column (default: 'elapsed_s')
    title     : overall figure title
    save_path : if given, save the figure to this path instead of showing it
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    x = data[x_col]
    n = len(columns)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]   # always a list

    fig.suptitle(title or "Telemetry", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, columns):
        if col not in data:
            ax.set_ylabel(col)
            ax.text(0.5, 0.5, f'"{col}" not in CSV', transform=ax.transAxes,
                    ha="center", va="center", color="red")
            continue
        ax.plot(x, data[col], linewidth=0.9)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)
        # zero-line for signed signals
        if any(v < 0 for v in data[col] if isinstance(v, float)):
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")

    axes[-1].set_xlabel(x_col)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[DataPlotter] Saved → {save_path}")
    else:
        plt.show()


# ── convenience wrapper ───────────────────────────────────────────────────────

def plot_run(columns: list, csv_path=None, x_col: str = "elapsed_s",
             title: str = "", save_path: str = None):
    """Load a run and plot selected columns in one call."""
    data = load(csv_path)
    run_name = Path(csv_path).parent.name if csv_path else latest_csv().parent.name
    plot_columns(data, columns,
                 x_col=x_col,
                 title=title or run_name,
                 save_path=save_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optionally pass a CSV path as the first argument
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # ── Edit this list to choose what to plot ─────────────────────────────────
    COLUMNS = [
        "speed_kmh",
        "accel_ms2",
        "steering",
        "throttle_brake",
    ]
    # ─────────────────────────────────────────────────────────────────────────

    plot_run(COLUMNS, csv_path=csv_arg)
