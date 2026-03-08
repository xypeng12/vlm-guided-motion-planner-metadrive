from plan.hard_feasible import min_clearance_to_points
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class Weights:
    w_efficiency: float = 0.9
    w_comfort: float = 0.7
    w_safety: float = 1.2

def curvature_cost(traj: np.ndarray) -> float:
    if traj is None or len(traj) < 5:
        return 0.0
    d1 = traj[1:] - traj[:-1]
    d2 = d1[1:] - d1[:-1]
    return float(np.mean(np.sum(d2 * d2, axis=1)))

def score_traj(
    traj: np.ndarray,
    w: Weights,
    summary: Dict[str, Any],
) -> float:
    efficiency = progress_score(traj)
    comfort = curvature_cost(traj)

    obs_pts = summary.get("obstacle_points_world", [])
    if obs_pts:
        clr = min_clearance_to_points(traj, np.asarray(obs_pts, np.float32))
    else:
        clr = 50.0

    lane_center_pen = float(summary.get("lane_center_penalty", 0.0))
    safety = float(np.tanh(clr / 5.0) - 0.25 * lane_center_pen)

    return (
        w.w_efficiency * efficiency
        - w.w_comfort * comfort
        + w.w_safety * safety
    )

def progress_score(traj: np.ndarray) -> float:
    if traj is None or len(traj) < 2:
        return 0.0
    return float(np.linalg.norm(traj[-1] - traj[0]))
