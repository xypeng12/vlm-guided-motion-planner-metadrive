from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from metadrive.envs import MetaDriveEnv

from utils import safe_getattr
from config import BLOCK_DETECT_DIST_M, FRONT_RANGE_M, LIDAR_MAX_RANGE_M


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
    """
    Extract the 240-ray lidar distance array from the MetaDrive obs vector.

    MetaDrive's LidarStateObservation packs the flat obs as:
        [ego_state(9) | navi_info(10) | num_others*4 surroundings | 240 lidar rays]

    With num_others=4 the layout is:
        indices 0–8    : ego state
        indices 9–18   : navigation info
        indices 19–34  : 4 closest vehicles × 4 dims (relative pos + vel)
        indices 35–274 : 240 lidar ray distances (normalised 0-1, 1=clear)

    We try named-key access first (dict obs), then fall back to slicing the
    flat array using the known offsets.
    """
    # ── named-key obs (some MetaDrive variants / wrappers use dicts) ──────────
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

    # ── flat numpy array (default MetaDrive obs) ──────────────────────────────
    # Layout with sensors configured in my_env.py vehicle_config:
    #   base ego state              :  6 dims  (heading, speed, steering, actions, yaw)
    #   side_detector cloud points  :  4 dims  (num_lasers=4, replaces 2-scalar default)
    #   lane_line_detector points   :  4 dims  (num_lasers=4, replaces 1-scalar default)
    #   navi info                   : 10 dims
    #   num_others surroundings     : 16 dims  (4 vehicles × 4 dims)
    #   lidar rays                  :240 dims
    #   Total                       :280 dims
    NUM_LIDAR_RAYS    = 240
    EGO_DIM           = 6 + 4 + 4   # base + side_detector(4) + lane_line_detector(4)
    NAVI_DIM          = 10
    NUM_OTHERS        = 4            # must match lidar.num_others in my_env.py
    OTHERS_DIM        = NUM_OTHERS * 4
    LIDAR_START       = EGO_DIM + NAVI_DIM + OTHERS_DIM   # = 40

    try:
        arr = np.asarray(obs, dtype=np.float32).ravel()
        if arr.size >= LIDAR_START + NUM_LIDAR_RAYS:
            return arr[LIDAR_START: LIDAR_START + NUM_LIDAR_RAYS]
    except Exception:
        pass

    return None


def block_ahead_from_lidar(
    lidar: Optional[np.ndarray],
    dist_th: float = 18.0,
    lateral_th: float = 2.5,
    max_range: float = LIDAR_MAX_RANGE_M,
) -> Tuple[bool, Optional[float]]:
    """Detect an obstacle in the forward cone of the lidar.

    MetaDrive's 240-ray lidar returns *normalised* distances in [0, 1]
    where 1.0 = no obstacle within ``max_range`` metres and 0.0 = contact.
    We convert to metres before comparing against ``dist_th``.
    """
    if lidar is None:
        return False, None

    x = np.asarray(lidar, dtype=np.float32)

    if x.ndim == 1 and x.size > 0:
        # Only look at the *forward* cone.  With 240 rays spanning 360°,
        # the forward ±20° sector is approximately rays 0-13 and 227-239.
        # This narrow cone avoids false positives from adjacent-lane traffic.
        n = x.size
        sector = max(1, n // 18)     # ±20° on each side of forward
        fwd_idx = np.concatenate([np.arange(0, sector), np.arange(n - sector, n)])
        fwd = x[fwd_idx] * max_range          # normalised → metres
        dmin = float(np.min(fwd))
        return (dmin < dist_th), dmin

    if x.ndim == 2 and x.shape[0] > 0 and x.shape[1] >= 2:
        forward = x[:, 0] * max_range
        lateral = x[:, 1] * max_range
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
                # Add front+rear points per vehicle for better collision geometry.
                # A car is ~4.5m long, ~2m wide. Use heading to project corners.
                tv_yaw = float(safe_getattr(tv, "heading_theta", 0.0))
                half_len = 2.2
                half_wid = 1.0
                cx, cy = np.cos(tv_yaw), np.sin(tv_yaw)
                for dl in (-half_len, 0.0, half_len):
                    for dw in (-half_wid, 0.0, half_wid):
                        ox = px + dl * cx - dw * cy
                        oy = py + dl * cy + dw * cx
                        obs_pts.append([ox, oy])
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

    # block detection: use traffic states with closing-speed filter as primary.
    # Only flag a vehicle as a "block" if it is both in the ego lane AND
    # the ego is closing on it (i.e. ego is faster, or it is stationary).
    if traffic_states and lane is not None:
        ego_fwd_speed = ego_speed  # scalar along heading
        c_ego = float(np.cos(ego_yaw))
        s_ego = float(np.sin(ego_yaw))
        try:
            s0, _ = lane.local_coordinates((ego_x, ego_y))
        except Exception:
            s0 = 0.0

        best_ds = None
        for st in traffic_states:
            px, py = st["x"], st["y"]
            if np.hypot(px - ego_x, py - ego_y) < 2.0:
                continue
            try:
                s_t, l_t = lane.local_coordinates((px, py))
            except Exception:
                continue
            ds = float(s_t - s0)
            if ds <= 0.0 or ds > FRONT_RANGE_M:
                continue
            if abs(float(l_t)) > 2.0:  # not in ego lane
                continue
            # closing speed: project target velocity along lane
            tv_long = st["vx"] * c_ego + st["vy"] * s_ego
            closing = ego_fwd_speed - tv_long  # positive = ego approaching
            # Only flag as block if closing > 1 m/s OR target is very slow
            if closing < 1.0 and tv_long > 3.0:
                continue
            if best_ds is None or ds < best_ds:
                best_ds = ds
        if best_ds is not None:
            summary["block_ahead"] = True
            summary["dist_to_block_m"] = float(best_ds)

    # Also check static objects on lane (no velocity → always a block)
    if not summary["block_ahead"] and pts_world is not None and len(pts_world) > 0:
        # Only use non-traffic obstacle points (static objects)
        traffic_xy = set()
        for st in traffic_states:
            traffic_xy.add((round(st["x"], 1), round(st["y"], 1)))
        static_pts = [p for p in obs_pts if (round(p[0], 1), round(p[1], 1)) not in traffic_xy]
        if static_pts:
            static_arr = np.asarray(static_pts, dtype=np.float32)
            ba_static, d_static = estimate_block_ahead_from_points_on_lane(
                env, static_arr, front_m=FRONT_RANGE_M, lane_half_width=2.0, ignore_radius_m=2.0
            )
            if ba_static:
                summary["block_ahead"] = True
                summary["dist_to_block_m"] = d_static

    # Emergency lidar fallback: very close obstacles (< 10m) not in traffic manager
    if not summary["block_ahead"]:
        lidar = get_lidar_from_obs(obs)
        block_lidar, dist_lidar = block_ahead_from_lidar(
            lidar, dist_th=min(10.0, BLOCK_DETECT_DIST_M), lateral_th=2.5
        )
        if block_lidar:
            summary["block_ahead"] = True
            summary["dist_to_block_m"] = dist_lidar

    # TTC rough + alongside-vehicle detection + target-lane forward gap
    min_ttc_left = None
    min_ttc_right = None
    # Also track if any vehicle is alongside (within longitudinal range) in
    # the adjacent lane — these have infinite TTC but are still dangerous.
    alongside_left = False
    alongside_right = False
    # Track the nearest vehicle AHEAD in the adjacent lane (any speed).
    # This catches slower vehicles 12-30 m ahead that slip through the
    # alongside window and the TTC filter (which requires closing speed).
    nearest_left_fwd = float("inf")
    nearest_right_fwd = float("inf")
    _LANE_LATERAL_LO = 1.5   # inner edge of adjacent-lane lateral band
    _LANE_LATERAL_HI = 6.0   # outer edge

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

        # Check for alongside vehicles: within ±8m longitudinally and in
        # the lateral band corresponding to an adjacent lane (~2.5-6m away).
        if -6.0 < forward < 12.0:
            if _LANE_LATERAL_LO < lateral < _LANE_LATERAL_HI:
                alongside_left = True
            elif -_LANE_LATERAL_HI < lateral < -_LANE_LATERAL_LO:
                alongside_right = True

        # Track nearest vehicle ahead in adjacent lane (regardless of speed)
        if 0.0 < forward < 35.0:
            if _LANE_LATERAL_LO < lateral < _LANE_LATERAL_HI:
                nearest_left_fwd = min(nearest_left_fwd, forward)
            elif -_LANE_LATERAL_HI < lateral < -_LANE_LATERAL_LO:
                nearest_right_fwd = min(nearest_right_fwd, forward)

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

    # Minimum forward clearance in target lane to allow a lane change.
    # Must be far enough that by the time ego completes the manoeuvre (~2-3 s)
    # it won't close the gap.  15 m ≈ 1.5 s at typical 10 m/s delta.
    _MIN_TARGET_LANE_GAP_M = 15.0

    if summary["has_left_lane"]:
        ttc_ok = (min_ttc_left is None) or (min_ttc_left > 3.0)
        fwd_ok = nearest_left_fwd > _MIN_TARGET_LANE_GAP_M
        summary["gap_left_ok"] = ttc_ok and (not alongside_left) and fwd_ok
    if summary["has_right_lane"]:
        ttc_ok = (min_ttc_right is None) or (min_ttc_right > 3.0)
        fwd_ok = nearest_right_fwd > _MIN_TARGET_LANE_GAP_M
        summary["gap_right_ok"] = ttc_ok and (not alongside_right) and fwd_ok

    return summary
