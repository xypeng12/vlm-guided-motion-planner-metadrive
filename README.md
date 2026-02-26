# Metadrive Motion Planning Demo
MAE252 – interactive MetaDrive sandbox for manual driving, map visualisation, and telemetry logging.

---

## Setup

```bash
conda activate metadrive
```

All scripts are run from the repo root after activating the environment.

---

## Files

| File | Purpose |
|---|---|
| `drive.py` | Open a 3-D window and drive manually |
| `draw_map.py` | Render a top-down 2-D PNG of any map |
| `logger.py` | Per-step telemetry logger (CSV + optional MP4) |
| `DataPlotter.py` | Plot telemetry columns from a saved CSV |
| `my_env.py` | Environment factory (`make_env`, `make_highway_env`) |

---

## drive.py  –  Manual driving

```bash
# Custom long highway (recommended starting point)
python drive.py --highway

# Highway with video recording saved to logs/
python drive.py --highway --record

# Highway with more lanes and heavier traffic
python drive.py --highway --lanes 4 --traffic 0.3

# Highway with higher incident probability
python drive.py --highway --incident 0.8

# Generic map presets
python drive.py --map roundabout
python drive.py --map mixed --traffic 0.2
python drive.py --map intersection --seed 5
```

**In-game controls**

| Key | Action |
|---|---|
| W / Up | Accelerate |
| S / Down | Brake / reverse |
| A / Left | Steer left |
| D / Right | Steer right |
| R | Reset scenario |
| Q | Quit |

**CLI flags**

| Flag | Default | Description |
|---|---|---|
| `--highway` | off | Launch the long custom highway (overrides `--map`) |
| `--map` | `mixed` | Generic map preset (see list below) |
| `--lanes` | 3 | Base lanes per road section |
| `--traffic` | 0.15 | NPC traffic density 0–1 |
| `--seed` | 0 | Scenario seed |
| `--incident` | 0.5 | [highway only] Roadside incident probability 0–1 |
| `--record` | off | Save an MP4 to the log folder |
| `--no-log` | off | Disable CSV telemetry logging entirely |
| `--run-name` | — | Optional label appended to the log folder name |
| `--debug` | off | Extra console output; press 1 to toggle physics colliders |

**Map presets** (pass to `--map`)

```
straight | curve | intersection | roundabout |
highway  | t_junction | mixed | random_3 | random_5
```

---

## draw_map.py  –  Top-down 2-D map

```bash
# Default: renders the custom highway map at seed 0
python draw_map.py

# Different seed
python draw_map.py --seed 5

# Generic map preset
python draw_map.py --map roundabout
python draw_map.py --map mixed --seed 3

# Higher resolution output
python draw_map.py --size 4096
```

Output is saved to `maps/map_<name>_seed<N>.png`.

**CLI flags**

| Flag | Default | Description |
|---|---|---|
| `--map` | — | Generic map preset; omit to draw the custom highway |
| `--seed` | 0 | Scenario seed |
| `--lanes` | 3 | Base lane count |
| `--size` | 2048 | Image resolution in pixels |
| `--out` | auto | Output filename (default: `map_<name>_seed<N>.png`) |

---

## DataPlotter.py  –  Telemetry plots

```bash
# Auto-load the most recent run and show plots
python DataPlotter.py

# Load a specific run
python DataPlotter.py logs/20260226_114659_mixed/telemetry.csv
```

Edit the `COLUMNS` list near the bottom of the file to choose which signals to plot.  
Available columns: `speed_kmh`, `accel_ms2`, `steering`, `throttle_brake`, `reward`, `cost`, `x`, `y`, `heading_deg`, `lane_changes`, `on_broken_line`, `crash_vehicle`, `crash_object`, and more.

---

## Logs

Each run writes to `logs/<timestamp>_<name>/`:

```
logs/
  20260226_114659_highway/
    telemetry.csv   # full time-series, one row per step
    summary.csv     # single-row run statistics
    run.mp4         # 3-D window video (only when --record is used)
```

The `logs/` folder is git-ignored.
