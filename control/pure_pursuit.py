from typing import List

import numpy as np

from metadrive.envs import MetaDriveEnv
from utils import safe_getattr, wrap_to_pi


# =========================
# Tracking controller
# =========================
def follow_polyline_pure_pursuit(
    env: MetaDriveEnv,
    path: np.ndarray,
    lookahead_m: float = 12.0,
    wheelbase_m: float = 2.7,
    v_ref: float = 20.0,
    steer_scale: float = 2.8,
) -> List[float]:
    v = env.vehicle
    if path is None or len(path) < 5:
        return [0.0, -0.2]

    ego = np.array([float(v.position[0]), float(v.position[1])], dtype=np.float32)
    ego_yaw = float(safe_getattr(v, "heading_theta", 0.0))
    speed = float(safe_getattr(v, "speed", 0.0))

    d = np.linalg.norm(path - ego[None, :], axis=1)
    i0 = int(np.argmin(d))

    dist = 0.0
    it = i0
    while it + 1 < len(path) and dist < lookahead_m:
        dist += float(np.linalg.norm(path[it + 1] - path[it]))
        it += 1
    tgt = path[it]

    dx, dy = float(tgt[0] - ego[0]), float(tgt[1] - ego[1])
    bearing = float(np.arctan2(dy, dx))
    alpha = wrap_to_pi(bearing - ego_yaw)

    Ld = max(float(np.hypot(dx, dy)), 1e-3)
    delta = float(np.arctan2(2.0 * wheelbase_m * np.sin(alpha), Ld))
    steer = float(np.clip(steer_scale * delta, -1.0, 1.0))

    turn_slow = 1.0 - 0.7 * min(1.0, abs(steer))
    v_target = v_ref * turn_slow
    u = float(np.clip((v_target - speed) / max(v_ref, 1e-3), -1.0, 1.0))

    return [steer, u]