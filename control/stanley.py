from typing import Tuple

from typing import List
import numpy as np
from metadrive.envs import MetaDriveEnv
from utils import safe_getattr

def follow_polyline_stanley(
    env: MetaDriveEnv,
    path: np.ndarray,
    v_ref: float = 20.0,
    wheelbase_m: float = 2.7,
    k_stanley: float = 1.2,
    k_soft: float = 2.0,
    steer_scale: float = 1.0,
    max_steer: float = 1.0,
) -> List[float]:
    """
    Stanley lateral control + simple speed P controller.
    Output: [steer, throttle] in [-1,1].
    """
    v = env.vehicle
    if path is None or len(path) < 5:
        return [0.0, -0.2]

    ego = np.array([float(v.position[0]), float(v.position[1])], dtype=np.float32)
    ego_yaw = float(safe_getattr(v, "heading_theta", 0.0))
    speed = float(safe_getattr(v, "speed", 0.0))

    # 1) nearest index on path
    i, _ = nearest_point_on_polyline(path, ego)

    # 2) path heading and heading error
    psi_path = heading_of_segment(path, i)
    e_psi = wrap_to_pi(psi_path - ego_yaw)

    # 3) signed cross-track error (CTE)
    # sign via path tangent normal
    p_ref = path[i]
    dx, dy = float(p_ref[0] - ego[0]), float(p_ref[1] - ego[1])
    # normal pointing left of path tangent
    nx, ny = -np.sin(psi_path), np.cos(psi_path)
    e_ct = dx * nx + dy * ny  # signed

    # 4) Stanley steering law
    # delta = heading_error + atan(k*cte / (v + soft))
    delta = e_psi + np.arctan2(k_stanley * e_ct, speed + k_soft)

    steer = steer_scale * delta
    steer = clamp(steer, -max_steer, max_steer)

    # 5) speed control (simple P)
    throttle = speed_p_controller(speed, v_ref=v_ref, v_scale=max(5.0, v_ref))

    return [steer, throttle]


def nearest_point_on_polyline(path_xy: np.ndarray, p_xy: np.ndarray) -> Tuple[int, float]:
    """
    Return (closest_index, closest_dist)
    """
    d = np.linalg.norm(path_xy - p_xy[None, :], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])

def heading_of_segment(path_xy: np.ndarray, i: int) -> float:
    """
    Heading of path at index i using forward difference.
    """
    n = len(path_xy)
    j = min(i + 1, n - 1)
    i0 = max(i - 1, 0)
    # prefer forward if possible
    if j != i:
        dx, dy = path_xy[j, 0] - path_xy[i, 0], path_xy[j, 1] - path_xy[i, 1]
    else:
        dx, dy = path_xy[i, 0] - path_xy[i0, 0], path_xy[i, 1] - path_xy[i0, 1]
    return float(np.arctan2(dy, dx))

def wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2*np.pi) - np.pi)

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def speed_p_controller(v: float, v_ref: float, v_scale: float = 20.0) -> float:
    """
    Very simple speed controller: output throttle/brake in [-1,1].
    """
    e = v_ref - v
    u = e / max(1e-3, v_scale)
    return clamp(u, -1.0, 1.0)