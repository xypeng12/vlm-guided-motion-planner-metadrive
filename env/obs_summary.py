from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from metadrive.envs import MetaDriveEnv

from utils import safe_getattr


def get_adjacent_lanes(lane, env: Optional[MetaDriveEnv] = None) -> Tuple[Optional[Any], Optional[Any]]:
    if lane is None:
        return None, None

    # Strict adjacent-lane path: use lane_id +/- 1 on the same (start,end) road key.
    # This prevents jumping to non-adjacent connected lanes.
    if env is not None:
        try:
            idx = safe_getattr(lane, "index", None)
            if idx is None:
                idx = safe_getattr(getattr(env, "vehicle", None), "lane_index", None)
            if isinstance(idx, tuple) and len(idx) == 3:
                start, end, lane_id = idx
                lane_id = int(lane_id)
                rn = safe_getattr(getattr(env, "current_map", None), "road_network", None)
                graph = safe_getattr(rn, "graph", None)
                if graph is not None and start in graph and end in graph[start]:
                    lanes = graph[start][end]
                    if isinstance(lanes, (list, tuple)) and len(lanes) > 0:
                        left_strict = lanes[lane_id + 1] if (lane_id + 1) < len(lanes) else None
                        right_strict = lanes[lane_id - 1] if (lane_id - 1) >= 0 else None

                        ego = safe_getattr(getattr(env, "vehicle", None), "position", None)
                        ego_xy = (float(ego[0]), float(ego[1])) if ego is not None else None

                        def _lat_side(cand):
                            if cand is None or ego_xy is None:
                                return None
                            try:
                                s_c, _ = cand.local_coordinates(ego_xy)
                                s_c = float(s_c)
                                c_len = safe_getattr(cand, "length", None)
                                if c_len is not None:
                                    s_c = float(np.clip(s_c, 0.0, max(0.0, float(c_len) - 1e-3)))
                                p = cand.position(s_c, 0.0)
                                _, lat = lane.local_coordinates((float(p[0]), float(p[1])))
                                return float(lat)
                            except Exception:
                                return None

                        left = left_strict
                        right = right_strict
                        lat_l = _lat_side(left_strict)
                        lat_r = _lat_side(right_strict)

                        # If index ordering and geometric side disagree, swap.
                        if lat_l is not None and lat_l < -1e-3 and lat_r is not None and lat_r > 1e-3:
                            left, right = right_strict, left_strict

                        return left, right
        except Exception:
            pass

    candidates: List[Any] = []

    def _add_candidate(obj) -> None:
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for o in obj:
                _add_candidate(o)
            return
        candidates.append(obj)

    def _lane_heading(lane_obj, s_val: float) -> Optional[float]:
        try:
            ds = 0.5
            p0 = lane_obj.position(float(s_val), 0.0)
            p1 = lane_obj.position(float(s_val + ds), 0.0)
            dx = float(p1[0]) - float(p0[0])
            dy = float(p1[1]) - float(p0[1])
            if abs(dx) + abs(dy) < 1e-6:
                return None
            return float(np.arctan2(dy, dx))
        except Exception:
            return None

    def _wrap_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    # Gather method-based neighbor hints.
    for name in [
        "left_lane", "get_left_lane", "left_neighbor", "get_left_neighbor", "left_lanes",
        "right_lane", "get_right_lane", "right_neighbor", "get_right_neighbor", "right_lanes",
    ]:
        if hasattr(lane, name):
            try:
                obj = getattr(lane, name)
                _add_candidate(obj() if callable(obj) else obj)
            except Exception:
                pass

    # Gather immediate neighbors from road graph by lane-id +/- 1 (strict adjacent).
    if env is not None:
        try:
            idx = safe_getattr(lane, "index", None)
            if idx is None:
                idx = safe_getattr(getattr(env, "vehicle", None), "lane_index", None)
            if isinstance(idx, tuple) and len(idx) == 3:
                start, end, lane_id = idx
                lane_id = int(lane_id)
                rn = safe_getattr(getattr(env, "current_map", None), "road_network", None)
                graph = safe_getattr(rn, "graph", None)
                if graph is not None and start in graph and end in graph[start]:
                    lanes = graph[start][end]
                    if isinstance(lanes, (list, tuple)):
                        if (lane_id - 1) >= 0:
                            _add_candidate(lanes[lane_id - 1])
                        if (lane_id + 1) < len(lanes):
                            _add_candidate(lanes[lane_id + 1])
        except Exception:
            pass

    # Deduplicate.
    uniq: List[Any] = []
    seen = set()
    for c in candidates:
        if c is None or c is lane:
            continue
        key = id(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    ego = safe_getattr(getattr(env, "vehicle", None), "position", None) if env is not None else None
    ego_xy = (float(ego[0]), float(ego[1])) if ego is not None else None

    s_base = 0.0
    if ego_xy is not None:
        try:
            s_base, _ = lane.local_coordinates(ego_xy)
            s_base = float(s_base)
        except Exception:
            s_base = 0.0
    base_h = _lane_heading(lane, s_base)

    left = None
    right = None
    left_abs = None
    right_abs = None

    for cand in uniq:
        try:
            if ego_xy is not None:
                s_c, _ = cand.local_coordinates(ego_xy)
                s_c = float(s_c)
            else:
                s_c = 0.0
            c_len = safe_getattr(cand, "length", None)
            if c_len is not None:
                s_c = float(np.clip(s_c, 0.0, max(0.0, float(c_len) - 1e-3)))
        except Exception:
            continue

        cand_h = _lane_heading(cand, s_c)
        if base_h is not None and cand_h is not None:
            dpsi = abs(_wrap_pi(cand_h - base_h))
            if dpsi > (np.pi / 2.0):
                continue

        try:
            p = cand.position(s_c, 0.0)
            _, lat = lane.local_coordinates((float(p[0]), float(p[1])))
            lat = float(lat)
        except Exception:
            continue

        if lat > 1e-3:
            if left_abs is None or abs(lat) < left_abs:
                left_abs = abs(lat)
                left = cand
        elif lat < -1e-3:
            if right_abs is None or abs(lat) < right_abs:
                right_abs = abs(lat)
                right = cand

    return left, right



def get_lidar_from_obs(obs: Any) -> Optional[np.ndarray]:
    if isinstance(obs, dict):
        for k in ["lidar", "Lidar", "lidar_state", "lidar_cloud", "point_cloud"]:
            if k in obs:
                try:
                    return np.asarray(obs[k])
                except Exception:
                    return None
        for _, v in obs.items():
            if isinstance(v, dict):
                for kk in ["lidar", "Lidar", "lidar_state", "lidar_cloud", "point_cloud"]:
                    if kk in v:
                        try:
                            return np.asarray(v[kk])
                        except Exception:
                            return None
    return None


def block_ahead_from_lidar(
    lidar: Optional[np.ndarray],
    dist_th: float = 18.0,
    lateral_th: float = 2.5,
) -> Tuple[bool, Optional[float]]:
    if lidar is None:
        return False, None

    x = np.asarray(lidar)

    if x.ndim == 1 and x.size > 0:
        dmin = float(np.min(x))
        return (dmin < dist_th), dmin

    if x.ndim == 2 and x.shape[0] > 0 and x.shape[1] >= 2:
        forward = x[:, 0]
        lateral = x[:, 1]
        mask = (forward > 0.0) & (np.abs(lateral) < lateral_th)
        if np.any(mask):
            dmin = float(np.min(forward[mask]))
            return (dmin < dist_th), dmin

    return False, None


def estimate_block_ahead_from_points_on_lane(
    env,
    pts_world: np.ndarray,
    front_m: float = 35.0,
    lane_half_width: float = 2.0,
    ignore_radius_m: float = 2.0,
) -> Tuple[bool, Optional[float]]:
    v = env.vehicle
    lane = safe_getattr(v, "lane", None)
    if lane is None or pts_world is None or len(pts_world) == 0:
        return False, None

    ego_x, ego_y = float(v.position[0]), float(v.position[1])
    try:
        s0, _ = lane.local_coordinates((ego_x, ego_y))
    except Exception:
        return False, None

    best_ds = None
    for p in pts_world:
        px, py = float(p[0]), float(p[1])
        if np.hypot(px - ego_x, py - ego_y) < ignore_radius_m:
            continue
        try:
            s, l = lane.local_coordinates((px, py))
        except Exception:
            continue
        ds = float(s - s0)
        if ds <= 0.0 or ds > front_m:
            continue
        if abs(float(l)) > lane_half_width:
            continue
        if best_ds is None or ds < best_ds:
            best_ds = ds

    if best_ds is None:
        return False, None
    return True, float(best_ds)



def build_state_summary(env: MetaDriveEnv, obs: Any) -> Dict[str, Any]:
    v = env.vehicle
    lane = safe_getattr(v, "lane", None)
    left_lane, right_lane = get_adjacent_lanes(lane, env) if lane is not None else (None, None)

    ego_x, ego_y = float(v.position[0]), float(v.position[1])
    ego_yaw = float(safe_getattr(v, "heading_theta", 0.0))
    ego_speed = float(safe_getattr(v, "speed", 0.0))

    summary: Dict[str, Any] = {
        "speed": ego_speed,
        "pos": [ego_x, ego_y],
        "yaw": ego_yaw,
        "has_left_lane": left_lane is not None,
        "has_right_lane": right_lane is not None,

        "block_ahead": False,
        "dist_to_block_m": None,

        "gap_left_ok": None,
        "gap_right_ok": None,
        "min_ttc_left": None,
        "min_ttc_right": None,

        "lane_center_penalty": 0.0,
        "right_bias_bonus": 0.0,

        "obstacle_points_world": [],
    }

    obs_pts: List[List[float]] = []
    traffic_states: List[Dict[str, float]] = []

    # traffic vehicles
    try:
        tm = getattr(env.engine, "traffic_manager", None)
        vehicles = None
        if tm is not None and hasattr(tm, "vehicles"):
            vehicles = tm.vehicles
            if isinstance(vehicles, dict):
                vehicles = list(vehicles.values())

        if vehicles is not None:
            for tv in vehicles:
                if tv is None or tv is v:
                    continue
                pos = safe_getattr(tv, "position", None)
                if pos is None:
                    continue
                px, py = float(pos[0]), float(pos[1])

                vel = safe_getattr(tv, "velocity", None)
                if vel is not None:
                    vx, vy = float(vel[0]), float(vel[1])
                else:
                    spd = float(safe_getattr(tv, "speed", 0.0))
                    yaw = float(safe_getattr(tv, "heading_theta", 0.0))
                    vx, vy = spd * np.cos(yaw), spd * np.sin(yaw)

                traffic_states.append({"x": px, "y": py, "vx": vx, "vy": vy})
                obs_pts.append([px, py])
    except Exception:
        pass

    # static objects
    try:
        om = getattr(env.engine, "object_manager", None)
        if om is not None:
            objs = None
            if hasattr(om, "get_objects"):
                try:
                    objs = om.get_objects()
                except Exception:
                    objs = None
            if objs is None and hasattr(om, "_object_registry"):
                try:
                    objs = getattr(om, "_object_registry")
                except Exception:
                    objs = None

            if isinstance(objs, dict):
                objs = list(objs.values())

            if objs is not None:
                for o in objs:
                    if o is None:
                        continue
                    pos = safe_getattr(o, "position", None)
                    if pos is None:
                        continue
                    obs_pts.append([float(pos[0]), float(pos[1])])
    except Exception:
        pass

    # dedup
    pts_world = np.zeros((0, 2), dtype=np.float32)
    if obs_pts:
        arr = np.asarray(obs_pts, dtype=np.float32)
        key = np.round(arr / 0.5)
        _, idx = np.unique(key, axis=0, return_index=True)
        pts_world = arr[np.sort(idx)]
        summary["obstacle_points_world"] = pts_world.tolist()

    # block detection: lidar first
    lidar = get_lidar_from_obs(obs)
    block_ahead, dist = block_ahead_from_lidar(lidar, dist_th=18.0, lateral_th=2.5)
    if block_ahead:
        summary["block_ahead"] = True
        summary["dist_to_block_m"] = dist
    else:
        if pts_world is not None and len(pts_world) > 0:
            ba2, d2 = estimate_block_ahead_from_points_on_lane(
                env, pts_world, front_m=35.0, lane_half_width=2.0, ignore_radius_m=2.0
            )
            summary["block_ahead"] = bool(ba2)
            summary["dist_to_block_m"] = d2

    # TTC rough
    min_ttc_left = None
    min_ttc_right = None

    c = float(np.cos(ego_yaw))
    s = float(np.sin(ego_yaw))

    def rel_long_lat(px, py):
        dx, dy = px - ego_x, py - ego_y
        forward = dx * c + dy * s
        lateral = -dx * s + dy * c
        return forward, lateral

    for st in traffic_states:
        px, py, vx, vy = st["x"], st["y"], st["vx"], st["vy"]
        forward, lateral = rel_long_lat(px, py)
        if forward <= 0.0 or forward > 40.0:
            continue

        rel_v_long = (vx - ego_speed * c) * c + (vy - ego_speed * s) * s
        if rel_v_long >= -1e-3:
            continue

        ttc = forward / (-rel_v_long + 1e-6)

        if lateral > 1.2:
            if min_ttc_left is None or ttc < min_ttc_left:
                min_ttc_left = float(ttc)
        elif lateral < -1.2:
            if min_ttc_right is None or ttc < min_ttc_right:
                min_ttc_right = float(ttc)

    summary["min_ttc_left"] = min_ttc_left
    summary["min_ttc_right"] = min_ttc_right

    if summary["has_left_lane"]:
        summary["gap_left_ok"] = (min_ttc_left is None) or (min_ttc_left > 3.0)
    if summary["has_right_lane"]:
        summary["gap_right_ok"] = (min_ttc_right is None) or (min_ttc_right > 3.0)

    return summary
