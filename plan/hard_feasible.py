import numpy as np
def hard_feasible(
    env,
    traj,
    summary,
    min_clearance_m: float = 1.8,
    front_range_m: float = 35.0,
    side_range_m: float = 8.0,
    ignore_prefix_m: float = 0.0,
):
    v = env.vehicle
    ego_xy = np.array([float(v.position[0]), float(v.position[1])], dtype=np.float32)
    ego_yaw = float(getattr(v, "heading_theta", 0.0))

    obs_pts = summary.get("obstacle_points_world", [])
    if obs_pts:
        pts = np.asarray(obs_pts, dtype=np.float32)
        pts_front = filter_obstacles_front_sector(
            ego_xy, ego_yaw, pts,
            front_range_m=front_range_m,
            side_range_m=side_range_m,
            ignore_radius_m=2.0,
        )
        cmin = min_clearance_to_points(traj, pts_front, ignore_prefix_m=ignore_prefix_m)
        if cmin < min_clearance_m:
            return False, f"collision_pred(cmin={cmin:.2f})"

    return True, "ok"

def filter_obstacles_front_sector(
    ego_xy: np.ndarray,
    ego_yaw: float,
    obs_pts: np.ndarray,
    front_range_m: float = 35.0,
    side_range_m: float = 8.0,
    ignore_radius_m: float = 2.0,
) -> np.ndarray:
    if obs_pts is None or len(obs_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    rel = obs_pts - ego_xy[None, :]
    dist = np.linalg.norm(rel, axis=1)

    keep = dist > ignore_radius_m
    rel = rel[keep]
    if len(rel) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    c = np.cos(ego_yaw)
    s = np.sin(ego_yaw)
    forward = rel[:, 0] * c + rel[:, 1] * s
    lateral = -rel[:, 0] * s + rel[:, 1] * c

    keep2 = (forward > 0.0) & (forward < front_range_m) & (np.abs(lateral) < side_range_m)
    rel2 = rel[keep2]
    if len(rel2) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return ego_xy[None, :] + rel2


def min_clearance_to_points(
    traj: np.ndarray,
    pts: np.ndarray,
    ignore_prefix_m: float = 0.0,
) -> float:
    if traj is None or len(traj) == 0 or pts is None or len(pts) == 0:
        return 1e9

    use_traj = np.asarray(traj, dtype=np.float32)
    if ignore_prefix_m > 1e-6 and len(use_traj) >= 2:
        seg = np.linalg.norm(use_traj[1:] - use_traj[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        keep = s >= float(ignore_prefix_m)
        if np.any(keep):
            use_traj = use_traj[keep]
        else:
            use_traj = use_traj[-1:]

    d = np.linalg.norm(use_traj[:, None, :] - pts[None, :, :], axis=2)
    return float(np.min(d))
