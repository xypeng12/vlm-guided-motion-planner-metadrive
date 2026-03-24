"""
MetaDrive 0.4.3
TAMP-lite (10Hz planner) + Qwen2-VL (1Hz) policy via Hugging Face on Mac (MPS/CPU)

Outputs TWO GIFs:
  1) CAM: driving view + prompt I/O overlay
  2) BEV: topdown view + candidate paths + prompt I/O overlay
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import imageio

from metadrive.envs import MetaDriveEnv

from visual import (
    get_topdown_frame,
    world_to_topdown_px,
    world_to_local_topdown_px,
    draw_ranked_candidates,
    _wrap_lines,
    draw_text_panel,
)
from utils import to_uint8, pick_device


from env.env import get_family_config, step_compat, warmup_render

from policy.VLM import Qwen2VLPolicy
from policy.simple_policy import SimpleTextHeuristicPolicy
from control.pure_pursuit import follow_polyline_pure_pursuit
from control.stanley import follow_polyline_stanley

from env.obs_summary import build_state_summary,safe_getattr
from plan.planner import plan_once
from plan.lane_cand import gen_keep_lane_candidates
from config import (
    USE_VLM,
    CONTROLLER,
    VLM_MODEL,
    VLM_USE_IMAGE,
    VLM_UPDATE_HZ,
    PRINT_EVERY,
    OUT_GIF_CAM,
    OUT_GIF_BEV,
    MANEUVERS,
    DT,
    DEFAULT_OBJECTIVE_WEIGHTS,
    HUMAN_INSTRUCTION,
    SCENARIO_FAMILY,
    GIF_OVERLAY_FONT_SIZE,
    GIF_OVERLAY_PANEL_WIDTH,
    GIF_OVERLAY_PANEL_HEIGHT,
    GIF_OVERLAY_MARGIN,
    GIF_CAND_LABEL_FONT_SIZE,
    GIF_CAND_LEGEND_FONT_SIZE,
)

# =========================
# Main run
# =========================
def run(
    env_cfg: Dict[str, Any],
    out_gif_cam: str,
    out_gif_bev: str,
    steps: int | None = None,
    fps: int = 20,
    instruction: str = HUMAN_INSTRUCTION,
    bev_size: Tuple[int, int] = (800, 800),
):
    os.makedirs(os.path.dirname(out_gif_cam) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_gif_bev) or ".", exist_ok=True)

    print("[INFO] device:", pick_device())
    print("[INFO] USE_VLM:", USE_VLM, "VLM_MODEL:", VLM_MODEL, "VLM_USE_IMAGE:", VLM_USE_IMAGE, "VLM_UPDATE_HZ:", VLM_UPDATE_HZ)

    env = MetaDriveEnv(dict(env_cfg))
    obs = env.reset()
    warmup_render(env, n=12)

    env_horizon = int(env_cfg.get("horizon", 0) or 0)
    if steps is None:
        if env_horizon > 0:
            max_steps = env_horizon + max(50, int(round(0.10 * env_horizon)))
        else:
            max_steps = 900
    else:
        max_steps = int(steps)
    print(f"[INFO] step budget: {max_steps} (env horizon: {env_horizon if env_horizon > 0 else 'unknown'})")

    # warm start:
    # map starts can place ego on a very short lane segment (~10m), which makes
    # long-horizon lane-change candidate generation fail immediately.
    for _ in range(60):
        obs, _, done, _ = step_compat(env, [0.0, 0.35])
        if done:
            obs = env.reset()
            continue
        lane = safe_getattr(env.vehicle, "lane", None)
        lane_len = safe_getattr(lane, "length", None)
        if lane_len is not None and float(lane_len) >= 25.0:
            break

    cam = env.engine.get_sensor("main_camera")

    frames_cam: List[np.ndarray] = []
    frames_bev: List[np.ndarray] = []

    policy = Qwen2VLPolicy(VLM_MODEL) if USE_VLM else SimpleTextHeuristicPolicy()

    policy_out = {
        "bias": {m: 0.0 for m in MANEUVERS},
        "weights": dict(DEFAULT_OBJECTIVE_WEIGHTS),
        "notes": "init",
        "_prompt": "",
        "_raw": "",
    }

    next_vlm_t = 0.0

    def _compact_text(x: Any, max_chars: int = 220) -> str:
        if x is None:
            return ""
        s = " ".join(str(x).split())
        return s if len(s) <= max_chars else (s[: max_chars - 3] + "...")

    def collision_detail(step_info: Dict[str, Any]) -> str:
        if not isinstance(step_info, dict):
            return ""
        tags: List[str] = []
        if bool(step_info.get("crash_vehicle", False)):
            tags.append("vehicle")
        if bool(step_info.get("crash_object", False)):
            tags.append("object")
        if bool(step_info.get("crash_building", False)):
            tags.append("building")
        if bool(step_info.get("crash_human", False)):
            tags.append("human")
        if bool(step_info.get("crash_sidewalk", False)):
            tags.append("sidewalk")
        if not tags and bool(step_info.get("crash", False)):
            tags.append("unknown")
        return ",".join(tags)

    def opposite_change(maneuver: str) -> str:
        if maneuver == "ChangeLaneLeft":
            return "ChangeLaneRight"
        if maneuver == "ChangeLaneRight":
            return "ChangeLaneLeft"
        return ""

    lane_change_lock_dir = ""
    lane_change_lock_steps = 0
    lane_change_lock_steps_cfg = int(max(1, round(1.6 / max(1e-6, DT))))

    lane_change_cooldown_steps = 0
    lane_change_cooldown_steps_cfg = int(max(1, round(3.0 / max(1e-6, DT))))
    last_completed_change = ""

    lane_index_prev = safe_getattr(env.vehicle, "lane_index", None)
    lane_id_prev = lane_index_prev[2] if isinstance(lane_index_prev, tuple) and len(lane_index_prev) >= 3 else None

    brake_streak = 0
    last_motion_traj: np.ndarray | None = None
    episode_done = False

    for k in range(max_steps):
        sim_t = k * DT

        summary = build_state_summary(env, obs)
        # Add maneuver hysteresis to avoid immediate opposite-direction lane changes.
        local_policy_out = dict(policy_out)
        local_bias = dict(policy_out.get("bias", {}))
        for mm in MANEUVERS:
            if mm not in local_bias:
                local_bias[mm] = 0.0

        if lane_change_lock_steps > 0 and lane_change_lock_dir in MANEUVERS:
            local_bias[lane_change_lock_dir] = float(local_bias.get(lane_change_lock_dir, 0.0)) + 3.0
            opp = opposite_change(lane_change_lock_dir)
            if opp in MANEUVERS:
                local_bias[opp] = float(local_bias.get(opp, 0.0)) - 4.0
            if "KeepLane" in MANEUVERS:
                local_bias["KeepLane"] = float(local_bias.get("KeepLane", 0.0)) - 0.6

        if lane_change_cooldown_steps > 0:
            if "KeepLane" in MANEUVERS:
                local_bias["KeepLane"] = float(local_bias.get("KeepLane", 0.0)) + 2.0
            opp2 = opposite_change(last_completed_change)
            if opp2 in MANEUVERS:
                local_bias[opp2] = float(local_bias.get(opp2, 0.0)) - 5.0
        if brake_streak >= 8:
            if "Brake" in MANEUVERS:
                local_bias["Brake"] = float(local_bias.get("Brake", 0.0)) - 6.0
            if "KeepLane" in MANEUVERS:
                local_bias["KeepLane"] = float(local_bias.get("KeepLane", 0.0)) + 2.0

        local_policy_out["bias"] = local_bias
        m, traj, reason, dbg = plan_once(env, obs, local_policy_out)

        if m in ("ChangeLaneLeft", "ChangeLaneRight") and lane_change_lock_steps <= 0:
            lane_change_lock_dir = m
            lane_change_lock_steps = lane_change_lock_steps_cfg

        # control
        if m == "Brake":
            brake_streak += 1
            crawl_traj = None
            try:
                keep_cands = gen_keep_lane_candidates(
                    env,
                    horizon_m=22.0,
                    ds_ctrl=4.0,
                    lateral_offsets=(0.0,),
                )
                if keep_cands:
                    crawl_traj = keep_cands[0]
            except Exception:
                crawl_traj = None

            if crawl_traj is None and last_motion_traj is not None and len(last_motion_traj) >= 5:
                crawl_traj = last_motion_traj

            if crawl_traj is not None and len(crawl_traj) >= 5:
                if CONTROLLER == "stanley":
                    action = follow_polyline_stanley(env, crawl_traj, v_ref=5.0)
                else:
                    action = follow_polyline_pure_pursuit(
                        env,
                        crawl_traj,
                        lookahead_m=7.0,
                        v_ref=5.0,
                        steer_scale=1.7,
                    )
                speed_now = float(safe_getattr(env.vehicle, "speed", 0.0))
                if speed_now < 4.0:
                    action[1] = float(np.clip(action[1], 0.10, 0.32))
                else:
                    action[1] = float(np.clip(action[1], -0.05, 0.20))
                last_motion_traj = np.asarray(crawl_traj, dtype=np.float32)
            else:
                action = [0.0, 0.15]
        else:
            brake_streak = 0
            if traj is not None and len(traj) >= 2:
                last_motion_traj = np.asarray(traj, dtype=np.float32)
            if CONTROLLER == "stanley":
                action = follow_polyline_stanley(env, traj, v_ref=(12.0 if m.startswith("ChangeLane") else 20.0))
            else:
                action = follow_polyline_pure_pursuit(
                    env,
                    traj,
                    lookahead_m=(8.0 if m.startswith("ChangeLane") else 14.0),
                    v_ref=(12.0 if m.startswith("ChangeLane") else 20.0),
                    steer_scale=(2.2 if m.startswith("ChangeLane") else 2.4),
                )

        obs, reward, done, info = step_compat(env, action)

        if lane_change_lock_steps > 0:
            lane_change_lock_steps -= 1
            if lane_change_lock_steps == 0:
                lane_change_lock_dir = ""

        lane_index_now = safe_getattr(env.vehicle, "lane_index", None)
        lane_id_now = lane_index_now[2] if isinstance(lane_index_now, tuple) and len(lane_index_now) >= 3 else None
        if lane_id_prev is not None and lane_id_now is not None and lane_id_now != lane_id_prev:
            if m in ("ChangeLaneLeft", "ChangeLaneRight"):
                last_completed_change = m
            else:
                last_completed_change = "ChangeLaneLeft" if int(lane_id_now) > int(lane_id_prev) else "ChangeLaneRight"
            lane_change_cooldown_steps = lane_change_cooldown_steps_cfg
            lane_change_lock_steps = 0
            lane_change_lock_dir = ""
        lane_id_prev = lane_id_now
        if lane_change_cooldown_steps > 0:
            lane_change_cooldown_steps -= 1

        # render tick
        env.engine.taskMgr.step()

        # --- main camera ---
        rgb_cam = None
        if cam is not None:
            raw = cam.perceive()
            rgb_cam = to_uint8(raw)
            if rgb_cam is not None:
                rgb_cam = rgb_cam[..., :3]

        # --- BEV via env.render(mode="topdown") ---
        # IMPORTANT: must call env.render(mode="topdown") to create top_down_renderer.
        rgb_bev = get_topdown_frame(env, size=bev_size)

        if k == 0:
            print("[DEBUG] rgb_bev is None?", rgb_bev is None,
                  "top_down_renderer exists?", getattr(env.engine, "top_down_renderer", None) is not None)

        # --- VLM update ---
        if (not USE_VLM and k == 0) or (USE_VLM and (k == 0 or sim_t >= next_vlm_t)):
            frame_for_vlm = rgb_cam if (VLM_USE_IMAGE and rgb_cam is not None) else None
            policy_out = policy(instruction, summary, frame_for_vlm)
            if USE_VLM:
                next_vlm_t += 1.0 / max(1e-6, VLM_UPDATE_HZ)

        # overlay lines
        b = policy_out.get("bias", {})
        w = policy_out.get("weights", {})
        notes = str(policy_out.get("notes", ""))[:160]

        speed = float(safe_getattr(env.vehicle, "speed", 0.0))
        crash_detail = collision_detail(info)
        lines: List[Any] = []
        if done and crash_detail:
            lines.append(f"TERMINAL=COLLISION ({crash_detail})")
        lines.append(f"behavior={m}")
        lines.append(f"speed={speed:.2f}")
        lines.append(f"efficiency={float(w.get('w_efficiency', 0.0)):.2f}")
        lines.append(f"comfort={float(w.get('w_comfort', 0.0)):.2f}")
        lines.append(f"safety={float(w.get('w_safety', 0.0)):.2f}")
        line_h_est = int(max(18, round(GIF_OVERLAY_FONT_SIZE * 1.25)))
        panel_h_need = int(max(GIF_OVERLAY_PANEL_HEIGHT, 2 * GIF_OVERLAY_MARGIN + line_h_est * len(lines)))

        # GIF#1 cam
        if rgb_cam is not None:
            panel_h_cam = int(max(1, min(panel_h_need, rgb_cam.shape[0] - 2 * GIF_OVERLAY_MARGIN)))
            panel_y_cam = max(0, rgb_cam.shape[0] - panel_h_cam - GIF_OVERLAY_MARGIN)
            frame1 = draw_text_panel(
                rgb_cam,
                lines,
                panel_w=GIF_OVERLAY_PANEL_WIDTH,
                panel_h=panel_h_cam,
                font_size=GIF_OVERLAY_FONT_SIZE,
                x=GIF_OVERLAY_MARGIN,
                y=panel_y_cam,
            )
            frames_cam.append(frame1)

        # GIF#2 bev + candidates
        if rgb_bev is not None:
            frame2 = rgb_bev

            # collect top candidates
            flat = []
            top = dbg.get("top", {}) if isinstance(dbg, dict) else {}
            vis_maneuvers = [m] if m in top else list(MANEUVERS)
            for mm in vis_maneuvers:
                for item in top.get(mm, []):
                    flat.append((mm, float(item.get("score", 0.0)), item.get("traj", None)))
            flat.sort(key=lambda x: x[1], reverse=True)
            flat = flat[:12]

            mapped_candidates: List[Dict[str, Any]] = []

            # prefer renderer mapping
            mapped_ok = False
            try:
                for (mm, sc, traj_xy) in flat:
                    if traj_xy is None:
                        continue
                    px = world_to_topdown_px(env, np.asarray(traj_xy))
                    if px is not None and len(px) >= 2:
                        mapped_candidates.append({"m": mm, "score": sc, "px": px})
                        mapped_ok = True
            except Exception:
                mapped_ok = False

            # fallback to local mapping if renderer mapping not available
            if not mapped_ok:
                try:
                    v = env.vehicle
                    ego_xy = np.array([float(v.position[0]), float(v.position[1])], dtype=np.float32)
                    H, W = frame2.shape[:2]
                    size = int(min(H, W))
                    meters = 80.0
                    for (mm, sc, traj_xy) in flat:
                        if traj_xy is None:
                            continue
                        px = world_to_local_topdown_px(np.asarray(traj_xy), ego_xy, size=size, meters=meters)
                        if px is not None and len(px) >= 2:
                            mapped_candidates.append({"m": mm, "score": sc, "px": px})
                except Exception:
                    pass

            # draw all candidates with clear top-3 highlights
            if mapped_candidates:
                frame2 = draw_ranked_candidates(
                    frame2,
                    mapped_candidates,
                    top_k_highlight=3,
                    label_font_size=GIF_CAND_LABEL_FONT_SIZE,
                    legend_font_size=GIF_CAND_LEGEND_FONT_SIZE,
                )

            panel_h_bev = int(max(1, min(panel_h_need, frame2.shape[0] - 2 * GIF_OVERLAY_MARGIN)))
            frame2 = draw_text_panel(
                frame2,
                lines,
                panel_w=GIF_OVERLAY_PANEL_WIDTH,
                panel_h=panel_h_bev,
                font_size=GIF_OVERLAY_FONT_SIZE,
                x=GIF_OVERLAY_MARGIN,
                y=max(0, frame2.shape[0] - panel_h_bev - GIF_OVERLAY_MARGIN),
            )
            frames_bev.append(frame2)

        stats = dbg.get("stats", {}) if isinstance(dbg, dict) else {}

        if k % PRINT_EVERY == 0:
            ba = summary.get("block_ahead", None)
            db = summary.get("dist_to_block_m", None)
            print(f"... block_ahead={ba} dist={None if db is None else round(db,2)}")
            if isinstance(stats, dict):
                for mm in ["KeepLane", "ChangeLaneLeft", "ChangeLaneRight"]:
                    s_mm = stats.get(mm, {})
                    reasons = s_mm.get("reasons", {})
                    if isinstance(reasons, dict) and len(reasons) > 0:
                        top_reason = max(reasons.items(), key=lambda kv: kv[1])
                    else:
                        top_reason = ("-", 0)
                    print(
                        f"... {mm}: gen={s_mm.get('generated', 0)} "
                        f"ok={s_mm.get('feasible', 0)} rej={s_mm.get('rejected', 0)} "
                        f"none={s_mm.get('no_candidates', 0)} "
                        f"min_cmin_rej={s_mm.get('min_cmin_reject', None)} "
                        f"top_rej={top_reason[0]} x{top_reason[1]}"
                    )
            print(f"... bias={ {mm: round(float(b.get(mm,0.0)),2) for mm in MANEUVERS} } notes={str(notes)[:90]}")
            print(f"k={k:04d} m={m:>14s} reason={reason:>10s} speed={speed:5.2f}")

        if done:
            if crash_detail:
                hold_frames = int(max(1, round(1.0 * fps)))
                if frames_cam:
                    frames_cam.extend([frames_cam[-1].copy() for _ in range(hold_frames)])
                if frames_bev:
                    frames_bev.extend([frames_bev[-1].copy() for _ in range(hold_frames)])
                print(f"[INFO] Collision terminal event: {crash_detail}; extended GIF by {hold_frames} frames.")
            print(f"[INFO] Episode done at step={k}, sim_t={sim_t:.2f}s, terminating run.")
            episode_done = True
            break

    if not episode_done:
        print(
            f"[INFO] Run reached configured steps={max_steps} before episode termination "
            f"(env horizon={env_cfg.get('horizon', 'unknown')})."
        )

    env.close()

    if frames_cam:
        imageio.mimsave(out_gif_cam, frames_cam, fps=fps)
        print(f"[DONE] Saved CAM GIF: {out_gif_cam} frames={len(frames_cam)} fps={fps}")
    else:
        print("[WARN] No main_camera frames captured.")

    if frames_bev:
        imageio.mimsave(out_gif_bev, frames_bev, fps=fps)
        print(f"[DONE] Saved BEV GIF: {out_gif_bev} frames={len(frames_bev)} fps={fps}")
    else:
        print("[WARN] No BEV frames captured. (topdown render failed?)")

if __name__ == "__main__":
    cfg = get_family_config(SCENARIO_FAMILY)

    run(
        cfg,
        out_gif_cam=OUT_GIF_CAM,
        out_gif_bev=OUT_GIF_BEV,
        steps=None,
        fps=20,
        instruction=HUMAN_INSTRUCTION,
        bev_size=(800, 800),
    )
