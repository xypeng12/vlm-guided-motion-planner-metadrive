from typing import List, Optional

import numpy as np
from metadrive.envs import MetaDriveEnv
from utils import wrap_to_pi
# -------------------------
# 3) Target point: current lane centerline, forward 10m
# -------------------------
def lane_forward_target(env: MetaDriveEnv, forward_m: float = 10.0) -> Optional[np.ndarray]:
    """
    Returns world target point [x, y] at lane centerline forward 10m.
    """
    veh = env.vehicle
    lane = getattr(veh, "lane", None)
    if lane is None:
        return None

    x, y = float(veh.position[0]), float(veh.position[1])
    s, _l = lane.local_coordinates((x, y))  # s: longitudinal
    s_tgt = float(s + forward_m)

    p = lane.position(s_tgt, 0.0)          # centerline: lateral=0
    return np.array([float(p[0]), float(p[1])], dtype=np.float32)

# -------------------------
# 4) Control policy: Pure Pursuit to that target + speed control
# -------------------------
def pure_pursuit_to_lane_target(
    env: MetaDriveEnv,
    forward_m: float = 10.0,
    wheelbase_m: float = 2.7,
    v_ref: float = 20.0,
    steer_scale: float = 1.0,
) -> List[float]:
    """
    Pure pursuit:
      delta = atan( 2L * sin(alpha) / lookahead )
    Here lookahead = forward_m (the target is defined that way).
    We then map delta to steer in [-1, 1] with a scale and clip.
    """
    veh = env.vehicle

    tgt = lane_forward_target(env, forward_m=forward_m)
    if tgt is None:
        # fallback: go straight, mild accel
        return [0.0, 0.2]

    # ego pose
    ego_x, ego_y = float(veh.position[0]), float(veh.position[1])
    ego_yaw = float(getattr(veh, "heading_theta", 0.0))

    # vector to target in world
    dx = float(tgt[0] - ego_x)
    dy = float(tgt[1] - ego_y)

    # bearing to target in world and relative angle in ego heading
    bearing = float(np.arctan2(dy, dx))
    alpha = wrap_to_pi(bearing - ego_yaw)

    # lookahead distance (approx; target is ~forward_m along lane, but still compute)
    Ld = float(np.hypot(dx, dy))
    Ld = max(Ld, 1e-3)

    # pure pursuit steering angle (radians)
    delta = float(np.arctan2(2.0 * wheelbase_m * np.sin(alpha), Ld))

    # map delta to steer control [-1, 1]
    # steer_scale is a tunable factor; you can increase if it understeers on sharp curves
    steer = float(np.clip(steer_scale * delta, -1.0, 1.0))

    # speed control (throttle/brake scalar)
    speed = float(getattr(veh, "speed", 0.0))

    # slow down on high steering
    turn_slow = 1.0 - 0.7 * min(1.0, abs(steer))
    v_target = v_ref * turn_slow

    u = float(np.clip((v_target - speed) / max(v_ref, 1e-3), -1.0, 1.0))

    return [steer, u]

