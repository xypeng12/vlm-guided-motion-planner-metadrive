from utils import safe_getattr
from typing import List, Tuple, Optional
from metadrive.envs import MetaDriveEnv
import numpy as np
from env.obs_summary import get_adjacent_lanes
from config import DEBUG_CANDIDATE_LOG, CURVE_GENERATION_METHOD
from plan.curve_generation import generate_curve

_DBG_PRINT_BUDGET = 60


def _dbg(msg: str) -> None:
    global _DBG_PRINT_BUDGET
    if not DEBUG_CANDIDATE_LOG:
        return
    if _DBG_PRINT_BUDGET <= 0:
        return
    print(msg)
    _DBG_PRINT_BUDGET -= 1


def _lane_id(lane) -> str:
    if lane is None:
        return "None"
    idx = safe_getattr(lane, "index", None)
    if idx is not None:
        return str(idx)
    idx2 = safe_getattr(lane, "lane_index", None)
    if idx2 is not None:
        return str(idx2)
    return f"<{lane.__class__.__name__}>"
# =========================
# B-spline (open uniform) evaluator
# =========================
def bspline_curve(control_pts: np.ndarray, n_samples: int = 70, degree: int = 3) -> np.ndarray:
    """
    Simple (not super optimized) open-uniform B-spline evaluation in 2D.
    control_pts: [M,2]
    returns: [n_samples,2]
    """
    P = np.asarray(control_pts, dtype=np.float32)
    M = P.shape[0]
    k = degree
    if M < k + 1:
        idx = np.linspace(0, M - 1, n_samples).astype(int)
        return P[idx]

    n_knots = M + k + 1
    knots = np.zeros(n_knots, dtype=np.float32)
    knots[k:M + 1] = np.linspace(0.0, 1.0, M - k + 1)
    knots[M + 1:] = 1.0

    # Cox–de Boor recursion
    def basis(i: int, kk: int, t: float) -> float:
        if kk == 0:
            return 1.0 if (knots[i] <= t < knots[i + 1] or (t == 1.0 and knots[i + 1] == 1.0)) else 0.0
        denom1 = knots[i + kk] - knots[i]
        denom2 = knots[i + kk + 1] - knots[i + 1]
        term1 = 0.0
        term2 = 0.0
        if denom1 > 1e-8:
            term1 = (t - knots[i]) / denom1 * basis(i, kk - 1, t)
        if denom2 > 1e-8:
            term2 = (knots[i + kk + 1] - t) / denom2 * basis(i + 1, kk - 1, t)
        return term1 + term2

    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    C = np.zeros((n_samples, 2), dtype=np.float32)
    for ti, t in enumerate(ts):
        pt = np.zeros(2, dtype=np.float32)
        wsum = 0.0
        for i in range(M):
            b = basis(i, k, float(t))
            wsum += b
            pt += b * P[i]
        if wsum > 1e-8:
            pt /= wsum
        C[ti] = pt
    return C

# =========================
# Trajectory candidates
# =========================
def gen_keep_lane_candidates(
    env: MetaDriveEnv,
    horizon_m: float = 45.0,
    ds_ctrl: float = 6.0,
    lateral_offsets: Tuple[float, ...] = (-0.35, 0.0, 0.35),
) -> List[np.ndarray]:
    v = env.vehicle
    lane = safe_getattr(v, "lane", None)
    s0 = lane_s_at_ego(env)
    if lane is None or s0 is None:
        return []

    ctrl_s = np.arange(s0, s0 + horizon_m + 1e-6, ds_ctrl, dtype=np.float32)
    cands = []
    for off in lateral_offsets:
        ctrl = sample_lane_points(lane, ctrl_s, lateral=off)
        traj = generate_curve(
            ctrl,
            method=CURVE_GENERATION_METHOD,
            n_samples=70,
            degree=3,
        )
        cands.append(traj)
    return cands

def lane_s_at_ego(env: MetaDriveEnv) -> Optional[float]:
    v = env.vehicle
    lane = safe_getattr(v, "lane", None)
    if lane is None:
        return None
    x, y = float(v.position[0]), float(v.position[1])
    try:
        s, _ = lane.local_coordinates((x, y))
        return float(s)
    except Exception:
        return None


def sample_lane_points(lane, s_list: np.ndarray, lateral: float) -> np.ndarray:
    pts = []
    for s in s_list:
        p = lane.position(float(s), float(lateral))
        pts.append([float(p[0]), float(p[1])])
    return np.array(pts, dtype=np.float32)


def _resample_polyline(ctrl_pts: np.ndarray, n_samples: int = 80) -> np.ndarray:
    pts = np.asarray(ctrl_pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    if len(pts) == 1:
        return np.repeat(pts, n_samples, axis=0)

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], n_samples, axis=0)

    q = np.linspace(0.0, total, n_samples, dtype=np.float32)
    out = np.zeros((n_samples, 2), dtype=np.float32)

    j = 0
    for i, qi in enumerate(q):
        while (j + 1) < len(s) and float(s[j + 1]) < float(qi):
            j += 1
        if (j + 1) >= len(s):
            out[i] = pts[-1]
            continue
        ds = float(s[j + 1] - s[j])
        if ds < 1e-6:
            out[i] = pts[j]
            continue
        u = float((qi - s[j]) / ds)
        out[i] = (1.0 - u) * pts[j] + u * pts[j + 1]
    return out


def _lane_side_vs_lane0(lane0, lane_cand, ego_xy: Tuple[float, float]) -> Optional[float]:
    """
    Signed lateral of candidate lane center measured in lane0 coordinates.
    Positive => candidate lane is on left side of lane0.
    """
    if lane0 is None or lane_cand is None:
        return None
    try:
        s_c, _ = lane_cand.local_coordinates((float(ego_xy[0]), float(ego_xy[1])))
        p = lane_cand.position(float(s_c), 0.0)
        _, lat = lane0.local_coordinates((float(p[0]), float(p[1])))
        return float(lat)
    except Exception:
        return None


def _adjacent_center_point_from_lane0_s(
    lane0,
    lane1,
    s0_ref: float,
    s1_ref: float,
    s_query: float,
    desired_side: int = 0,  # +1 left, -1 right
    s1_prev: Optional[float] = None,
    s_query_prev: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build lane0/lane1 center points at (approximately) the same station.
    Prefer projection from lane0 world point onto lane1; fallback to delta-s mapping.
    """
    p0w = lane0.position(float(s_query), 0.0)
    p0 = np.array([float(p0w[0]), float(p0w[1])], dtype=np.float32)

    if s1_prev is not None and s_query_prev is not None:
        s1_guess = float(s1_prev + (float(s_query) - float(s_query_prev)))
    else:
        s1_guess = float(s1_ref + (float(s_query) - float(s0_ref)))
    s1 = s1_guess
    try:
        s1_proj, _ = lane1.local_coordinates((float(p0[0]), float(p0[1])))
        if np.isfinite(s1_proj):
            s1_proj = float(s1_proj)
            # Keep projection locally consistent; avoid long jumps to unrelated stations.
            if s_query_prev is not None:
                ds_loc = max(0.6, abs(float(s_query) - float(s_query_prev)))
                lo = s1_guess - 1.5 * ds_loc
                hi = s1_guess + 1.5 * ds_loc
                s1_proj = float(np.clip(s1_proj, lo, hi))
                s1 = float(0.65 * s1_proj + 0.35 * s1_guess)
            else:
                s1 = s1_proj
    except Exception:
        s1 = s1_guess

    # Keep target lane station monotonic along control-point progression.
    if s1_prev is not None:
        s1 = max(float(s1_prev), float(s1))

    lane1_len = safe_getattr(lane1, "length", None)
    if lane1_len is not None:
        s1 = float(np.clip(s1, 0.0, max(0.0, float(lane1_len) - 1e-3)))

    def _p1_and_lat(s1_val: float) -> Tuple[np.ndarray, Optional[float]]:
        p1w_i = lane1.position(float(s1_val), 0.0)
        p1_i = np.array([float(p1w_i[0]), float(p1w_i[1])], dtype=np.float32)
        try:
            _, lat_i = lane0.local_coordinates((float(p1_i[0]), float(p1_i[1])))
            return p1_i, float(lat_i)
        except Exception:
            return p1_i, None

    p1, lat_sel = _p1_and_lat(s1)
    if desired_side != 0 and lat_sel is not None:
        wrong_side = (desired_side > 0 and lat_sel < -1e-3) or (desired_side < 0 and lat_sel > 1e-3)
        if wrong_side:
            p1_guess, lat_guess = _p1_and_lat(s1_guess)
            if lat_guess is not None:
                guess_ok = (desired_side > 0 and lat_guess > 1e-3) or (desired_side < 0 and lat_guess < -1e-3)
                if guess_ok:
                    s1 = float(s1_guess)
                    p1 = p1_guess
    return p0, p1, s1

def gen_change_lane_candidates(
    env: MetaDriveEnv,
    direction: str,  # "left" or "right"
    horizon_m: float = 55.0,
    ds_ctrl: float = 3.0,
    blend: Tuple[float, float, float] = (0.15, 0.45, 0.40),
    aggressiveness: Tuple[float, ...] = (1.0, 0.8),
) -> List[np.ndarray]:
    v = env.vehicle
    lane0 = safe_getattr(v, "lane", None)
    s0 = lane_s_at_ego(env)
    if lane0 is None or s0 is None:
        _dbg(f"[CAND] {direction}: early-exit lane0/s0 invalid lane0={lane0 is not None} s0={s0}")
        return []

    ego_xy = (float(v.position[0]), float(v.position[1]))

    left_lane, right_lane = get_adjacent_lanes(lane0, env)
    lane1 = left_lane if direction == "left" else right_lane
    if lane1 is None:
        _dbg(
            f"[CAND] {direction}: no adjacent lane "
            f"lane0={_lane_id(lane0)} left={_lane_id(left_lane)} right={_lane_id(right_lane)}"
        )
        return []

    # Side-sign sanity: ensure selected lane matches requested direction.
    lat_sel = _lane_side_vs_lane0(lane0, lane1, ego_xy)
    if direction == "left" and lat_sel is not None and lat_sel < 0.0 and right_lane is not None:
        lat_alt = _lane_side_vs_lane0(lane0, right_lane, ego_xy)
        if lat_alt is not None and lat_alt > 0.0:
            lane1 = right_lane
    if direction == "right" and lat_sel is not None and lat_sel > 0.0 and left_lane is not None:
        lat_alt = _lane_side_vs_lane0(lane0, left_lane, ego_xy)
        if lat_alt is not None and lat_alt < 0.0:
            lane1 = left_lane

    try:
        s1_ego, _ = lane1.local_coordinates(ego_xy)
        s1_ego = float(s1_ego)
    except Exception:
        # Fallback: if projection fails, use current-lane s as rough proxy.
        s1_ego = float(s0)
        _dbg(
            f"[CAND] {direction}: lane1 local_coordinates failed, fallback s1_ego=s0={s0:.2f} "
            f"lane0={_lane_id(lane0)} lane1={_lane_id(lane1)}"
        )

    # Debug lane-side sign check at ego station: positive=left of current lane, negative=right.
    try:
        p1e = lane1.position(float(s1_ego), 0.0)
        _, lat_chk = lane0.local_coordinates((float(p1e[0]), float(p1e[1])))
        _dbg(
            f"[CAND] {direction}: lane-side-check lane1_lat_vs_lane0={float(lat_chk):.3f} "
            f"lane0={_lane_id(lane0)} lane1={_lane_id(lane1)}"
        )
    except Exception:
        pass

    frac1, frac2, _frac3 = blend

    # Avoid sampling beyond lane ends where position(s, ...) may saturate and
    # produce degenerate short trajectories.
    lane0_len = safe_getattr(lane0, "length", None)
    lane1_len = safe_getattr(lane1, "length", None)

    max_h0 = float(horizon_m)
    max_h1 = float(horizon_m)
    if lane0_len is not None:
        max_h0 = max(0.0, float(lane0_len) - 1.0 - float(s0))
    if lane1_len is not None:
        max_h1 = max(0.0, float(lane1_len) - 1.0 - float(s1_ego))

    horizon_eff = min(float(horizon_m), max_h0, max_h1)
    if horizon_eff <= 0.8:
        _dbg(
            f"[CAND] {direction}: horizon too short horizon_eff={horizon_eff:.2f} ds={ds_ctrl:.2f} "
            f"s0={s0:.2f} s1_ego={s1_ego:.2f} len0={lane0_len} len1={lane1_len}"
        )
        return []

    s_end = s0 + horizon_eff

    s_mid1 = s0 + horizon_eff * frac1
    s_mid2 = s_mid1 + horizon_eff * frac2
    s_mid1 = min(s_mid1, s_end)
    s_mid2 = min(max(s_mid2, s_mid1 + 1e-3), s_end)

    # Short lane sections are common at map starts; adapt spacing instead of dropping all candidates.
    ds_eff = float(min(float(ds_ctrl), max(0.8, horizon_eff / 5.0)))
    ctrl_s = np.arange(s0, s_end + 1e-6, ds_eff, dtype=np.float32)
    if ctrl_s.size < 2:
        ctrl_s = np.asarray([s0, s_end], dtype=np.float32)

    cands = []
    for aggr in aggressiveness:
        ctrl = []
        s1_prev = None
        s_prev = None
        desired_side = 1 if direction == "left" else -1
        for s in ctrl_s:
            p0, p1, s1_prev = _adjacent_center_point_from_lane0_s(
                lane0=lane0,
                lane1=lane1,
                s0_ref=float(s0),
                s1_ref=float(s1_ego),
                s_query=float(s),
                desired_side=desired_side,
                s1_prev=s1_prev,
                s_query_prev=s_prev,
            )
            s_prev = float(s)

            # Strict terminal target on adjacent lane center (no projection drift).
            s1_nom = float(s1_ego + (float(s) - float(s0)))
            if lane1_len is not None:
                s1_nom = float(np.clip(s1_nom, 0.0, max(0.0, float(lane1_len) - 1e-3)))
            p1_nom_w = lane1.position(float(s1_nom), 0.0)
            p1_nom = np.array([float(p1_nom_w[0]), float(p1_nom_w[1])], dtype=np.float32)

            if s <= s_mid1:
                ctrl.append([float(p0[0]), float(p0[1])])
            elif s >= s_mid2:
                ctrl.append([float(p1_nom[0]), float(p1_nom[1])])
            else:
                u = (float(s) - float(s_mid1)) / max(1e-6, float(s_mid2 - s_mid1))
                # aggr < 1 => slower transition, aggr > 1 => faster transition
                u = np.clip(u * max(1e-6, aggr), 0.0, 1.0)
                # Smooth blend to avoid piecewise-kink near lane switch.
                u = u * u * (3.0 - 2.0 * u)
                px = (1 - u) * float(p0[0]) + u * float(p1_nom[0])
                py = (1 - u) * float(p0[1]) + u * float(p1_nom[1])
                ctrl.append([px, py])

        ctrl = np.array(ctrl, dtype=np.float32)
        traj = generate_curve(
            ctrl,
            method=CURVE_GENERATION_METHOD,
            n_samples=80,
            degree=3,
        )
        cands.append(traj)
    _dbg(
        f"[CAND] {direction}: generated={len(cands)} "
        f"lane0={_lane_id(lane0)} lane1={_lane_id(lane1)} s0={s0:.2f} s1_ego={s1_ego:.2f} "
        f"h_eff={horizon_eff:.2f} ds_eff={ds_eff:.2f}"
    )
    return cands
