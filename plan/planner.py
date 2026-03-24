from plan.lane_cand import gen_keep_lane_candidates, gen_change_lane_candidates
from env.obs_summary import build_state_summary
from typing import Any, Dict, Optional, Tuple

import numpy as np

from metadrive.envs import MetaDriveEnv
from plan.score import score_traj, Weights
from plan.hard_feasible import hard_feasible
from config import MANEUVERS, DEBUG_DISABLE_FEASIBILITY, MIN_CLEARANCE_M, FRONT_RANGE_M, BLOCK_DETECT_DIST_M, KEEP_LANE_BIAS, DEFAULT_OBJECTIVE_WEIGHTS


def _keep_lane_center_penalty(env: MetaDriveEnv, traj: np.ndarray, sample_n: int = 12) -> float:
    """
    Mean absolute lateral offset of the trajectory w.r.t. current lane center.
    Lower is better (more stable lane keeping).
    """
    if traj is None or len(traj) == 0:
        return 0.0
    lane = getattr(env.vehicle, "lane", None)
    if lane is None:
        return 0.0

    pts = np.asarray(traj, dtype=np.float32)
    if len(pts) > sample_n:
        idx = np.linspace(0, len(pts) - 1, sample_n).astype(int)
        pts = pts[idx]

    lats = []
    for p in pts:
        try:
            _, lat = lane.local_coordinates((float(p[0]), float(p[1])))
            lats.append(abs(float(lat)))
        except Exception:
            continue
    if not lats:
        return 0.0
    return float(np.mean(lats))

# Planning once
# =========================
def plan_once(env: MetaDriveEnv, obs: Any, policy_out: Dict[str, Any], use_vlm: bool = False) -> Tuple[str, Optional[np.ndarray], str, Dict[str, Any]]:
    summary = build_state_summary(env, obs)

    # If a lane-change lock is active, bypass the gap gate for that direction
    # so the car can complete a committed lane change.
    lock_dir = policy_out.get("lane_change_lock_dir", "")

    wj = policy_out.get("weights", {})
    w = Weights(
        w_efficiency=float(wj.get("w_efficiency", wj.get("w_progress", DEFAULT_OBJECTIVE_WEIGHTS["w_efficiency"]))),
        w_comfort=float(wj.get("w_comfort", DEFAULT_OBJECTIVE_WEIGHTS["w_comfort"])),
        w_safety=float(wj.get("w_safety", wj.get("w_clearance", DEFAULT_OBJECTIVE_WEIGHTS["w_safety"]))),
    )

    bias = policy_out.get("bias", {})
    rank = sorted(MANEUVERS, key=lambda mm: float(bias.get(mm, 0.0)), reverse=True)

    block = bool(summary.get("block_ahead", False))
    dist = summary.get("dist_to_block_m", None)
    gapL = summary.get("gap_left_ok", None)
    gapR = summary.get("gap_right_ok", None)

    def rule_bias(m: str) -> float:
        # When VLM is active, the VLM already receives the full road state
        # (block_ahead, dist_to_block, gaps) and outputs appropriate bias.
        # Applying rule_bias on top would partially override VLM decisions.
        if use_vlm:
            return 0.0
        b = 0.0
        if block and dist is not None:
            if dist < BLOCK_DETECT_DIST_M * 0.7:   # ~21 m at 30 m detect
                if m == "KeepLane":
                    b -= 2.0
                if m == "Brake":
                    b += 0.5
                if m == "ChangeLaneLeft" and (gapL is True):
                    b += 1.5
                if m == "ChangeLaneRight" and (gapR is True):
                    b += 1.5
            if dist < 8.0:
                if m == "Brake":
                    b += 2.0
                # Do NOT penalize lane changes at close range — if a lane
                # change is already in progress it must commit, not abort.
        return b

    dbg = {
        "top": {m: [] for m in MANEUVERS},
        "stats": {
            m: {
                "generated": 0,
                "feasible": 0,
                "rejected": 0,
                "no_candidates": 0,
                "reasons": {},
                "min_cmin_reject": None,
            }
            for m in MANEUVERS
        },
    }
    TOPK = 5

    best_score = -1e18
    best_traj = None
    best_m = "Brake"
    best_reason = "no_candidate"

    for m in rank:
        if m == "KeepLane":
            cands = gen_keep_lane_candidates(env)
        elif m == "ChangeLaneLeft":
            if gapL is False and lock_dir != "ChangeLaneLeft":
                dbg["stats"][m]["no_candidates"] += 1
                dbg["stats"][m]["reasons"]["gap_unsafe"] = 1
                continue
            cands = gen_change_lane_candidates(env, direction="left")
        elif m == "ChangeLaneRight":
            if gapR is False and lock_dir != "ChangeLaneRight":
                dbg["stats"][m]["no_candidates"] += 1
                dbg["stats"][m]["reasons"]["gap_unsafe"] = 1
                continue
            cands = gen_change_lane_candidates(env, direction="right")
        elif m == "Brake":
            cands = []
        else:
            cands = []

        if m != "Brake" and len(cands) == 0:
            dbg["stats"][m]["no_candidates"] += 1

        if m == "Brake":
            brake_score = float(bias.get("Brake", 0.0)) + rule_bias("Brake")
            had_traj = best_traj is not None
            if brake_score > best_score or not had_traj:
                best_m = "Brake"
                best_traj = None
                best_reason = "bias" if had_traj else "fallback"
                best_score = brake_score
            continue

        for traj in cands:
            dbg["stats"][m]["generated"] += 1
            if DEBUG_DISABLE_FEASIBILITY:
                ok, fail_reason = True, "debug_bypass"
            else:
                if m.startswith("ChangeLane"):
                    ok, fail_reason = hard_feasible(
                        env,
                        traj,
                        summary,
                        min_clearance_m=MIN_CLEARANCE_M + 0.7,
                        front_range_m=FRONT_RANGE_M,
                        side_range_m=10.0,
                        ignore_prefix_m=4.0,
                    )
                else:
                    ok, fail_reason = hard_feasible(
                        env, traj, summary,
                        min_clearance_m=MIN_CLEARANCE_M,
                        front_range_m=FRONT_RANGE_M,
                        side_range_m=4.5,
                    )
            if not ok:
                dbg["stats"][m]["rejected"] += 1
                reasons = dbg["stats"][m]["reasons"]
                reasons[fail_reason] = int(reasons.get(fail_reason, 0)) + 1
                if isinstance(fail_reason, str) and fail_reason.startswith("collision_pred(cmin="):
                    try:
                        cmin_val = float(fail_reason.split("cmin=")[1].rstrip(")"))
                        cur = dbg["stats"][m]["min_cmin_reject"]
                        if cur is None or cmin_val < float(cur):
                            dbg["stats"][m]["min_cmin_reject"] = cmin_val
                    except Exception:
                        pass
                continue

            dbg["stats"][m]["feasible"] += 1
            if DEBUG_DISABLE_FEASIBILITY:
                reasons = dbg["stats"][m]["reasons"]
                reasons["debug_bypass"] = int(reasons.get("debug_bypass", 0)) + 1
            sc = score_traj(traj, w, summary)
            if m == "KeepLane":
                # Prefer centerline keep-lane candidate to avoid lateral drift/oscillation.
                sc -= 0.8 * w.w_safety * _keep_lane_center_penalty(env, traj)
                # Flat preference for staying in lane — overcome only by explicit bias or
                # a large scoring difference (applies equally to VLM and baseline modes).
                sc += KEEP_LANE_BIAS
            sc += float(bias.get(m, 0.0))
            sc += rule_bias(m)

            dbg["top"][m].append({"score": float(sc), "traj": traj})
            dbg["top"][m].sort(key=lambda x: x["score"], reverse=True)
            dbg["top"][m] = dbg["top"][m][:TOPK]

            if sc > best_score:
                best_score = sc
                best_traj = traj
                best_m = m
                best_reason = "ok"

    return best_m, best_traj, best_reason, dbg
