# VLM-Guided Motion Planner (MetaDrive)

Lightweight trajectory planning stack for MetaDrive 0.4.3 with:
- candidate generation (`KeepLane`, `ChangeLaneLeft`, `ChangeLaneRight`, `Brake`)
- hard feasibility filtering
- multi-objective scoring (`efficiency`, `comfort`, `safety`)
- optional VLM-based preference/weight adaptation
- GIF visualization for camera view and BEV candidate trajectories

This README documents the current planner codebase only.

## Project Layout

```text
.
├── main.py                  # main simulation loop, planner-policy-control integration
├── config.py                # all runtime configuration
├── env/
│   ├── env.py               # scenario family configs (baseline/blocked/interactive)
│   └── obs_summary.py       # environment summary features for policy + planner
├── plan/
│   ├── planner.py           # choose best feasible trajectory
│   ├── lane_cand.py         # keep-lane and lane-change candidate generation
│   ├── hard_feasible.py     # hard collision/clearance feasibility checks
│   ├── score.py             # multi-objective trajectory score
│   └── curve_generation.py  # b-spline and clothoid curve backends
├── policy/
│   ├── VLM.py               # Qwen2-VL based policy
│   └── simple_policy.py     # fixed-weight fallback policy
├── control/
│   ├── pure_pursuit.py
│   └── stanley.py
├── visual.py                # BEV/candidate/text overlay drawing
└── save_gif/                # generated outputs
```

## Requirements

- Python 3.10+
- MetaDrive 0.4.3
- PyTorch
- transformers
- numpy
- Pillow
- imageio

Example install:

```bash
pip install metadrive==0.4.3 torch transformers numpy pillow imageio
```

## Quick Start

Run one episode and save two GIFs:

```bash
python main.py
```

Outputs are written to:
- `save_gif/metadrive_cam_prompt.gif`
- `save_gif/metadrive_bev_cand_prompt.gif`

## Main Configuration (`config.py`)

Key settings:

- `USE_VLM`: `True` to use Qwen2-VL policy, `False` for fixed simple policy
- `VLM_MODEL`: model id (default: `Qwen/Qwen2-VL-2B-Instruct`)
- `HUMAN_INSTRUCTION`: natural-language instruction fed to policy
- `DEFAULT_OBJECTIVE_WEIGHTS`: default `{w_efficiency, w_comfort, w_safety}`
- `SIMPLE_POLICY_WEIGHTS`: weights used when `USE_VLM=False`
- `CURVE_GENERATION_METHOD`: `"b_spline"` or `"clothoid"`
- `SCENARIO_FAMILY`: `"baseline"`, `"blocked"`, or `"interactive"`
- `CONTROLLER`: set `"stanley"` to use Stanley controller; otherwise pure pursuit path tracking is used

Visualization settings:
- `GIF_OVERLAY_FONT_SIZE`
- `GIF_OVERLAY_PANEL_WIDTH`
- `GIF_OVERLAY_PANEL_HEIGHT`
- `GIF_OVERLAY_MARGIN`
- `GIF_CAND_LABEL_FONT_SIZE`
- `GIF_CAND_LEGEND_FONT_SIZE`

## Runtime Flow

1. Build scenario config from `env/env.py`
2. Generate candidates for each maneuver
3. Run hard feasibility filtering
4. Score feasible candidates with weighted objective terms
5. Pick best-scoring feasible trajectory (fallback to `Brake` if none)
6. Track trajectory with pure-pursuit or Stanley
7. Render and save camera/BEV GIFs

## Policy Modes

### 1) VLM mode (`USE_VLM=True`)
- `policy/VLM.py` queries the VLM at `VLM_UPDATE_HZ`
- VLM returns JSON with:
  - maneuver bias values
  - objective weights (`efficiency`, `comfort`, `safety`)

### 2) Simple mode (`USE_VLM=False`)
- `policy/simple_policy.py` returns fixed biases/weights from config

## Notes

- First VLM run may download model weights from Hugging Face.
- If lane-change candidates seem too conservative, inspect:
  - `DEBUG_CANDIDATE_LOG`
  - `DEBUG_DISABLE_FEASIBILITY`
  - feasibility settings in `plan/hard_feasible.py` and `plan/planner.py`
- `save_gif/` is output data and can be excluded from commits.

