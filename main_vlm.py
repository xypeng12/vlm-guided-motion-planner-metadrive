"""
main_vlm.py  –  VLM-guided motion planner on map "SSrSRSSrSS"
=============================================================
Integrates the teammate's TAMP-lite planner + optional Qwen2-VL policy with
the project's existing MetaDrive setup.

Map: SSrSRSSrSS  (2 straight -> on-ramp -> straight -> off-ramp -> 2 straight -> on-ramp -> 2 straight)
  S = Straight   r = InRamp (merge on)   R = OutRamp (merge off)

Outputs:
  save_gif/metadrive_cam_prompt.gif   –  driver's camera view
  save_gif/metadrive_bev_cand_prompt.gif  –  top-down BEV with candidate paths

Usage:
    python main_vlm.py                   # no VLM (fast, no GPU needed)
    python main_vlm.py --vlm             # enable Qwen2-VL (needs transformers)
    python main_vlm.py --steps 300       # limit to N planner steps
    python main_vlm.py --controller stanley
    python main_vlm.py --no-gif          # skip saving GIFs (faster preview)
"""

import os
import sys
import argparse
import threading
from typing import Any, Dict, List, Tuple, Optional


# ── Fix: the workspace has a local metadrive/ git-clone directory that Python
# treats as a namespace package, shadowing the real installed package.
# We locate the installed package via CONDA_PREFIX and insert its path first.
def _fix_metadrive_sys_path() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return
    site = os.path.join(conda_prefix, "Lib", "site-packages")
    if not os.path.isdir(site):
        return

    def _try_add(target: str) -> bool:
        target = target.strip()
        if os.path.isdir(os.path.join(target, "metadrive")) and target not in sys.path:
            sys.path.insert(0, target)
            return True
        return False

    # 1) Any *metadrive*.egg-link (editable install via setup.py)
    for fname in os.listdir(site):
        if fname.endswith(".egg-link") and "metadrive" in fname.lower():
            try:
                with open(os.path.join(site, fname)) as f:
                    if _try_add(f.readline()):
                        return
            except Exception:
                continue

    # 2) Scan all .pth files (covers easy-install.pth and other editable installs)
    for fname in os.listdir(site):
        if not fname.endswith(".pth"):
            continue
        try:
            with open(os.path.join(site, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("import"):
                        if _try_add(line):
                            return
        except Exception:
            continue

_fix_metadrive_sys_path()

import numpy as np
import imageio

# Import torch BEFORE MetaDrive so its CUDA DLLs load first.
# MetaDrive (Panda3D) otherwise loads conflicting DLLs that prevent
# torch from initialising its CUDA extensions.
try:
    import torch as _torch_preload  # noqa: F401
except Exception:
    pass

from metadrive.envs import MetaDriveEnv

# ── project modules ──────────────────────────────────────────────────────────
from visual import (
    get_topdown_frame,
    world_to_topdown_px,
    world_to_local_topdown_px,
    draw_ranked_candidates,
    _wrap_lines,
    draw_text_panel,
)
from utils import to_uint8, pick_device, safe_getattr

from env.env import step_compat, warmup_render
from env.obs_summary import build_state_summary

from policy.VLM import Qwen2VLPolicy
from policy.simple_policy import SimpleTextHeuristicPolicy

from control.pure_pursuit import follow_polyline_pure_pursuit
from control.stanley import follow_polyline_stanley

from plan.planner import plan_once
from plan.lane_cand import gen_keep_lane_candidates

from logger import RunLogger

from config import (
    USE_VLM,
    CONTROLLER,
    VLM_MODEL,
    GIF_CAPTURE_EVERY,
    VLM_USE_IMAGE,
    VLM_UPDATE_HZ,
    PRINT_EVERY,
    OUT_GIF_CAM,
    OUT_GIF_BEV,
    MANEUVERS,
    DT,
    DEFAULT_OBJECTIVE_WEIGHTS,
    HUMAN_INSTRUCTION,
    MAP_SEQUENCE,
    GIF_OVERLAY_FONT_SIZE,
    GIF_OVERLAY_PANEL_WIDTH,
    GIF_OVERLAY_PANEL_HEIGHT,
    GIF_OVERLAY_MARGIN,
    GIF_CAND_LABEL_FONT_SIZE,
    GIF_CAND_LEGEND_FONT_SIZE,
    REPLAN_EVERY_STEPS,
)


class VLMRunLogger(RunLogger):
    """
    Extends RunLogger with extra columns for the VLM planner:
      maneuver, reason, block_ahead, dist_to_block_m, vlm_notes
    """
    EXTRA_COLUMNS = ["maneuver", "reason", "block_ahead", "dist_to_block_m", "vlm_notes", "w_efficiency", "w_comfort", "w_safety", "bias_KeepLane", "bias_ChangeLaneLeft", "bias_ChangeLaneRight", "bias_Brake"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Patch the CSV writer to include extra columns
        import csv
        all_cols = self.COLUMNS + self.EXTRA_COLUMNS
        self._file.seek(0)
        self._file.truncate()
        self._writer = csv.DictWriter(self._file, fieldnames=all_cols, extrasaction="ignore")
        self._writer.writeheader()

    def record_vlm(self, env, reward: float = 0.0, info: dict = None,
                   maneuver: str = "", reason: str = "",
                   block_ahead: bool = False, dist_to_block_m=None,
                   vlm_notes: str = "",
                   w_efficiency: float = 0.0, w_comfort: float = 0.0, w_safety: float = 0.0,
                   bias: dict = None):
        """Call instead of record() each step to capture planner state too."""
        # Temporarily patch env.agent → env.vehicle for compatibility
        _had_agent = hasattr(env, "agent")
        if not _had_agent:
            env.agent = env.vehicle
        try:
            self.record(env, reward=reward, info=info or {})
        finally:
            if not _had_agent:
                del env.agent

        # Append extra columns by re-reading and patching last row isn't ideal;
        # instead write them as a follow-up via direct writer access.
        import csv as _csv
        extra = {
            "maneuver":        maneuver,
            "reason":          reason,
            "block_ahead":     int(bool(block_ahead)),
            "dist_to_block_m": round(float(dist_to_block_m), 2) if dist_to_block_m is not None else "",
            "vlm_notes":       str(vlm_notes)[:120],
            "w_efficiency":    round(float(w_efficiency), 4),
            "w_comfort":       round(float(w_comfort), 4),
            "w_safety":        round(float(w_safety), 4),
            "bias_KeepLane":          round(float((bias or {}).get("KeepLane", 0.0)), 2),
            "bias_ChangeLaneLeft":    round(float((bias or {}).get("ChangeLaneLeft", 0.0)), 2),
            "bias_ChangeLaneRight":   round(float((bias or {}).get("ChangeLaneRight", 0.0)), 2),
            "bias_Brake":             round(float((bias or {}).get("Brake", 0.0)), 2),
        }
        # Patch: rewrite the last CSV line to include extra fields.
        # Simpler: keep our own side-file for extra columns appended per step.
        if not hasattr(self, "_extra_file"):
            import pathlib
            extra_path = pathlib.Path(self.run_dir) / "planner.csv"
            self._extra_file = open(extra_path, "w", newline="")
            self._extra_writer = _csv.DictWriter(
                self._extra_file,
                fieldnames=["timestep"] + self.EXTRA_COLUMNS
            )
            self._extra_writer.writeheader()
            print(f"[Logger] Planner log -> {extra_path}")

        try:
            step = env.engine.episode_step
        except Exception:
            step = ""
        self._extra_writer.writerow({"timestep": step, **extra})

    def close(self, summary: bool = True):
        if hasattr(self, "_extra_file"):
            self._extra_file.flush()
            self._extra_file.close()
        super().close(summary=summary)


# ── Async VLM wrapper ────────────────────────────────────────────────────────
class AsyncVLMWrapper:
    """
    Runs VLM inference in a background thread so the simulation loop
    (and Panda3D window) never freezes waiting for the model.
    Returns the last completed result while a new one is in flight.
    """
    def __init__(self, policy):
        self._policy = policy
        self._result: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._busy = False

    def trigger(self, instruction: str, summary: dict, frame_rgb) -> bool:
        """Start a background VLM call. Returns True if started, False if busy."""
        if self._busy:
            return False
        self._busy = True
        t = threading.Thread(
            target=self._run,
            args=(instruction, summary, frame_rgb),
            daemon=True,
        )
        t.start()
        return True

    def _run(self, instruction, summary, frame_rgb):
        try:
            result = self._policy(instruction, summary, frame_rgb)
        except Exception as e:
            print(f"[AsyncVLM] exception: {e}")
            result = None
        # Ensure result is always a non-empty dict so policy.ready becomes True.
        # An empty dict keeps ready=False and policy_out never updates.
        if not result:
            from config import DEFAULT_OBJECTIVE_WEIGHTS
            from plan.planner import MANEUVERS as _M
            result = {
                "bias": {m: 0.0 for m in _M},
                "weights": dict(DEFAULT_OBJECTIVE_WEIGHTS),
                "notes": "async_exception_defaults",
            }
        with self._lock:
            self._result = result
        self._busy = False

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._result)

    @property
    def ready(self) -> bool:
        return bool(self._result)


# ── env factory – delegates to my_env.py ────────────────────────────────────
from my_env import make_env as _make_env, MAP_PRESETS
from config import SCENARIO_SEED, RANDOM_TRAFFIC, TRAFFIC_DENSITY

def make_vlm_env(map_preset: str = MAP_SEQUENCE,
                 traffic_density: float = TRAFFIC_DENSITY,
                 num_lanes: int = 3,
                 seed: int = SCENARIO_SEED,
                 random_traffic: bool = RANDOM_TRAFFIC):
    """
    Build a MetaDriveEnv via my_env.make_env().
    Defaults come from config.py (TRAFFIC_DENSITY, SCENARIO_SEED, RANDOM_TRAFFIC).
    map_preset can be any key in my_env.MAP_PRESETS or a raw block string.
    """
    return _make_env(
        map_preset=map_preset,
        num_lanes=num_lanes,
        traffic_density=traffic_density,
        seed=seed,
        random_traffic=random_traffic,
        use_render=True,
        manual_control=False,   # planner drives; set True only for manual mode
    )


# ── main runner ──────────────────────────────────────────────────────────────
def run(
    env,                          # already-constructed MetaDriveEnv
    out_gif_cam: str,
    out_gif_bev: str,
    steps: Optional[int] = None,
    fps: int = 20,
    instruction: str = HUMAN_INSTRUCTION,
    bev_size: Tuple[int, int] = (800, 800),
    use_vlm: bool = USE_VLM,
    controller: str = CONTROLLER,
    save_gifs: bool = True,
    log: bool = True,
    run_name: str = "vlm",
):
    if save_gifs:
        os.makedirs(os.path.dirname(out_gif_cam) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(out_gif_bev) or ".", exist_ok=True)

    map_info = getattr(env, 'config', {}).get('map', '?')
    print(f"[INFO] Map          : {map_info}")
    print(f"[INFO] device       : {pick_device()}")
    print(f"[INFO] USE_VLM      : {use_vlm}  |  controller: {controller}")

    logger = VLMRunLogger(run_name=run_name) if log else None

    obs = env.reset()
    warmup_render(env, n=12)

    env_horizon = int((env.config.get("horizon") or 0))
    if steps is None:
        max_steps = (env_horizon + max(50, int(round(0.10 * env_horizon)))) if env_horizon > 0 else 1800
    else:
        max_steps = int(steps)
    print(f"[INFO] step budget  : {max_steps}")

    # warm-start: drive a bit so the ego is fully on-road before planning
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

    _base_policy = Qwen2VLPolicy(VLM_MODEL) if use_vlm else SimpleTextHeuristicPolicy()
    # Wrap VLM in async thread so inference never freezes the Panda3D window.
    # For the simple heuristic policy (instant), async is a no-op in practice.
    policy = AsyncVLMWrapper(_base_policy) if use_vlm else _base_policy

    policy_out: Dict[str, Any] = {
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

    lane_change_lock_dir      = ""
    lane_change_lock_steps    = 0
    lane_change_lock_steps_cfg = int(max(1, round(3.0 / max(1e-6, DT))))

    lane_change_cooldown_steps     = 0
    lane_change_cooldown_steps_cfg = int(max(1, round(3.0 / max(1e-6, DT))))
    last_completed_change = ""

    lane_index_prev = safe_getattr(env.vehicle, "lane_index", None)
    lane_id_prev    = lane_index_prev[2] if isinstance(lane_index_prev, tuple) and len(lane_index_prev) >= 3 else None

    brake_streak      = 0
    last_motion_traj: Optional[np.ndarray] = None
    episode_done      = False

    # ── replan interval state ─────────────────────────────────────────────────
    replan_countdown = 0          # steps until next replan; 0 = replan now
    held_m:      str               = "KeepLane"
    held_traj:   Optional[np.ndarray] = None
    held_reason: str               = "init"
    held_dbg:    Dict[str, Any]    = {}

    for k in range(max_steps):
        sim_t = k * DT

        summary = build_state_summary(env, obs)

        # ── decide whether to replan this step ────────────────────────────────
        # Force replan on first step, when countdown expires, or when there's
        # no usable trajectory and no Brake maneuver.
        should_replan = (
            replan_countdown <= 0
            or (held_traj is None and held_m != "Brake")
        )

        if should_replan:
            # maneuver hysteresis biases
            local_policy_out = dict(policy_out)
            local_bias = dict(policy_out.get("bias", {}))
            for mm in MANEUVERS:
                if mm not in local_bias:
                    local_bias[mm] = 0.0

            # ── hard lane-change lock ─────────────────────────────────────────
            # During the lock period, make the locked direction overwhelmingly
            # preferred so the planner commits to completing the lane change.
            if lane_change_lock_steps > 0 and lane_change_lock_dir in MANEUVERS:
                local_bias[lane_change_lock_dir] = float(local_bias.get(lane_change_lock_dir, 0.0)) + 20.0
                opp = opposite_change(lane_change_lock_dir)
                if opp in MANEUVERS:
                    local_bias[opp] = float(local_bias.get(opp, 0.0)) - 20.0
                if "KeepLane" in MANEUVERS:
                    local_bias["KeepLane"] = float(local_bias.get("KeepLane", 0.0)) - 10.0

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
            local_policy_out["lane_change_lock_dir"] = lane_change_lock_dir if lane_change_lock_steps > 0 else ""
            m, traj, reason, dbg = plan_once(env, obs, local_policy_out, use_vlm=use_vlm)

            # Start a new lane-change lock when a lane change is first chosen
            if m in ("ChangeLaneLeft", "ChangeLaneRight") and lane_change_lock_steps <= 0:
                lane_change_lock_dir   = m
                lane_change_lock_steps = lane_change_lock_steps_cfg

            # Save held plan and reset countdown
            held_m, held_traj, held_reason, held_dbg = m, traj, reason, dbg
            replan_countdown = REPLAN_EVERY_STEPS
        else:
            # Reuse last plan
            m, traj, reason, dbg = held_m, held_traj, held_reason, held_dbg
            replan_countdown -= 1

        # ── control ──────────────────────────────────────────────────────────
        if m == "Brake":
            brake_streak += 1
            crawl_traj = None
            try:
                keep_cands = gen_keep_lane_candidates(env, horizon_m=22.0, ds_ctrl=4.0, lateral_offsets=(0.0,))
                if keep_cands:
                    crawl_traj = keep_cands[0]
            except Exception:
                crawl_traj = None

            if crawl_traj is None and last_motion_traj is not None and len(last_motion_traj) >= 5:
                crawl_traj = last_motion_traj

            if crawl_traj is not None and len(crawl_traj) >= 5:
                if controller == "stanley":
                    action = follow_polyline_stanley(env, crawl_traj, v_ref=5.0)
                else:
                    action = follow_polyline_pure_pursuit(env, crawl_traj, lookahead_m=7.0, v_ref=5.0, steer_scale=1.7)
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
            if controller == "stanley":
                action = follow_polyline_stanley(
                    env, traj,
                    v_ref=(12.0 if m.startswith("ChangeLane") else 20.0),
                )
            else:
                action = follow_polyline_pure_pursuit(
                    env, traj,
                    lookahead_m=(8.0  if m.startswith("ChangeLane") else 14.0),
                    v_ref     =(12.0  if m.startswith("ChangeLane") else 20.0),
                    steer_scale=(2.2  if m.startswith("ChangeLane") else 2.4),
                )

        obs, reward, done, info = step_compat(env, action)

        # ── log telemetry ─────────────────────────────────────────────────────
        if logger is not None:
            ba  = summary.get("block_ahead", False)
            db  = summary.get("dist_to_block_m", None)
            notes = str(policy_out.get("notes", ""))[:120]
            _w = policy_out.get("weights", {})
            _b = policy_out.get("bias", {})
            logger.record_vlm(
                env, reward=reward, info=info,
                maneuver=m, reason=reason,
                block_ahead=ba, dist_to_block_m=db,
                vlm_notes=notes,
                w_efficiency=float(_w.get("w_efficiency", 0.0)),
                w_comfort=float(_w.get("w_comfort", 0.0)),
                w_safety=float(_w.get("w_safety", 0.0)),
                bias=_b,
            )

        # ── lane-change tracking ──────────────────────────────────────────────
        if lane_change_lock_steps > 0:
            lane_change_lock_steps -= 1
            if lane_change_lock_steps == 0:
                lane_change_lock_dir = ""

        lane_index_now = safe_getattr(env.vehicle, "lane_index", None)
        lane_id_now    = lane_index_now[2] if isinstance(lane_index_now, tuple) and len(lane_index_now) >= 3 else None
        if lane_id_prev is not None and lane_id_now is not None and lane_id_now != lane_id_prev:
            last_completed_change = (
                m if m in ("ChangeLaneLeft", "ChangeLaneRight")
                else ("ChangeLaneLeft" if int(lane_id_now) > int(lane_id_prev) else "ChangeLaneRight")
            )
            lane_change_cooldown_steps = lane_change_cooldown_steps_cfg
            lane_change_lock_steps     = 0
            lane_change_lock_dir       = ""
        lane_id_prev = lane_id_now
        if lane_change_cooldown_steps > 0:
            lane_change_cooldown_steps -= 1

        # ── render tick ───────────────────────────────────────────────────────
        env.engine.taskMgr.step()

        _capture_this_frame = save_gifs and (k % GIF_CAPTURE_EVERY == 0)
        _vlm_fires_this_step = (not use_vlm and k == 0) or (use_vlm and (k == 0 or sim_t >= next_vlm_t))

        # camera frame (only render when needed for GIF or VLM)
        rgb_cam = None
        if cam is not None and (_capture_this_frame or _vlm_fires_this_step):
            raw    = cam.perceive()
            rgb_cam = to_uint8(raw)
            if rgb_cam is not None:
                rgb_cam = rgb_cam[..., :3]

        # BEV frame (only render when needed for GIF)
        rgb_bev = get_topdown_frame(env, size=bev_size) if _capture_this_frame else None

        # ── VLM update (non-blocking) ─────────────────────────────────────────
        if _vlm_fires_this_step:
            frame_for_vlm = rgb_cam if (VLM_USE_IMAGE and rgb_cam is not None) else None
            if use_vlm:
                # Only advance schedule when inference actually starts (not when busy)
                if policy.trigger(instruction, summary, frame_for_vlm):
                    next_vlm_t += 1.0 / max(1e-6, VLM_UPDATE_HZ)
            else:
                policy_out = policy(instruction, summary, frame_for_vlm)

        # Pull latest completed VLM result (non-blocking)
        if use_vlm and policy.ready:
            latest = policy.get()
            if latest:
                policy_out = latest

        # ── overlay lines ─────────────────────────────────────────────────────
        b = policy_out.get("bias", {})
        w = policy_out.get("weights", {})
        speed       = float(safe_getattr(env.vehicle, "speed", 0.0))
        crash_detail = collision_detail(info)

        lines: List[Any] = []
        if done and crash_detail:
            lines.append(f"TERMINAL=COLLISION ({crash_detail})")
        lines.append(f"behavior={m}")
        lines.append(f"speed={speed:.2f}")
        lines.append(f"efficiency={float(w.get('w_efficiency', 0.0)):.2f}")
        lines.append(f"comfort={float(w.get('w_comfort', 0.0)):.2f}")
        lines.append(f"safety={float(w.get('w_safety', 0.0)):.2f}")

        line_h_est  = int(max(18, round(GIF_OVERLAY_FONT_SIZE * 1.25)))
        panel_h_need = int(max(GIF_OVERLAY_PANEL_HEIGHT, 2 * GIF_OVERLAY_MARGIN + line_h_est * len(lines)))

        if _capture_this_frame:
            # GIF #1 – camera
            if rgb_cam is not None:
                panel_h_cam = int(max(1, min(panel_h_need, rgb_cam.shape[0] - 2 * GIF_OVERLAY_MARGIN)))
                panel_y_cam = max(0, rgb_cam.shape[0] - panel_h_cam - GIF_OVERLAY_MARGIN)
                frame1 = draw_text_panel(
                    rgb_cam, lines,
                    panel_w=GIF_OVERLAY_PANEL_WIDTH,
                    panel_h=panel_h_cam,
                    font_size=GIF_OVERLAY_FONT_SIZE,
                    x=GIF_OVERLAY_MARGIN, y=panel_y_cam,
                )
                frames_cam.append(frame1)

            # GIF #2 – BEV + candidates
            if rgb_bev is not None:
                frame2 = rgb_bev
                flat = []
                top  = dbg.get("top", {}) if isinstance(dbg, dict) else {}
                for mm in (MANEUVERS if m not in top else [m]):
                    for item in top.get(mm, []):
                        flat.append((mm, float(item.get("score", 0.0)), item.get("traj", None)))
                flat.sort(key=lambda x: x[1], reverse=True)
                flat = flat[:12]

                mapped_candidates: List[Dict[str, Any]] = []
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

                if not mapped_ok:
                    try:
                        v      = env.vehicle
                        ego_xy = np.array([float(v.position[0]), float(v.position[1])], dtype=np.float32)
                        H, W   = frame2.shape[:2]
                        size   = int(min(H, W))
                        for (mm, sc, traj_xy) in flat:
                            if traj_xy is None:
                                continue
                            px = world_to_local_topdown_px(np.asarray(traj_xy), ego_xy, size=size, meters=80.0)
                            if px is not None and len(px) >= 2:
                                mapped_candidates.append({"m": mm, "score": sc, "px": px})
                    except Exception:
                        pass

                if mapped_candidates:
                    frame2 = draw_ranked_candidates(
                        frame2, mapped_candidates,
                        top_k_highlight=3,
                        label_font_size=GIF_CAND_LABEL_FONT_SIZE,
                        legend_font_size=GIF_CAND_LEGEND_FONT_SIZE,
                    )

                panel_h_bev = int(max(1, min(panel_h_need, frame2.shape[0] - 2 * GIF_OVERLAY_MARGIN)))
                frame2 = draw_text_panel(
                    frame2, lines,
                    panel_w=GIF_OVERLAY_PANEL_WIDTH,
                    panel_h=panel_h_bev,
                    font_size=GIF_OVERLAY_FONT_SIZE,
                    x=GIF_OVERLAY_MARGIN,
                    y=max(0, frame2.shape[0] - panel_h_bev - GIF_OVERLAY_MARGIN),
                )
                frames_bev.append(frame2)

        # ── stats print ───────────────────────────────────────────────────────
        stats = dbg.get("stats", {}) if isinstance(dbg, dict) else {}
        if k % PRINT_EVERY == 0:
            ba = summary.get("block_ahead", None)
            db = summary.get("dist_to_block_m", None)
            print(f"  block_ahead={ba} dist={None if db is None else round(db, 2)}")
            print(f"  k={k:04d} m={m:>14s} reason={reason:>10s} speed={speed:5.2f}")
            for mm in MANEUVERS:
                st = stats.get(mm, {})
                gen = st.get('generated', 0)
                fea = st.get('feasible', 0)
                rej = st.get('rejected', 0)
                noc = st.get('no_candidates', 0)
                rr = st.get('reasons', {})
                cmin = st.get('min_cmin_reject', None)
                if gen > 0 or noc > 0:
                    print(f"    {mm:20s} gen={gen} feas={fea} rej={rej} noc={noc} cmin={cmin} reasons={rr}")

        if done:
            if crash_detail:
                hold = int(max(1, round(1.0 * fps)))
                if frames_cam:
                    frames_cam.extend([frames_cam[-1].copy() for _ in range(hold)])
                if frames_bev:
                    frames_bev.extend([frames_bev[-1].copy() for _ in range(hold)])
                print(f"[INFO] Collision: {crash_detail}; extended GIF.")
            print(f"[INFO] Episode done at step={k}, sim_t={sim_t:.2f}s")
            episode_done = True
            break

    if not episode_done:
        print(f"[INFO] Reached budget steps={max_steps}")

    env.close()
    if logger is not None:
        logger.close()

    if save_gifs:
        if frames_cam:
            imageio.mimsave(out_gif_cam, frames_cam, fps=fps)
            print(f"[DONE] CAM GIF : {out_gif_cam}  ({len(frames_cam)} frames)")
        else:
            print("[WARN] No camera frames.")

        if frames_bev:
            imageio.mimsave(out_gif_bev, frames_bev, fps=fps)
            print(f"[DONE] BEV GIF : {out_gif_bev}  ({len(frames_bev)} frames)")
        else:
            print("[WARN] No BEV frames.")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _preset_choices = list(MAP_PRESETS.keys())

    parser = argparse.ArgumentParser(
        description="VLM-guided planner – uses my_env.py map presets",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--vlm",        action="store_true", help="Enable Qwen2-VL policy")
    parser.add_argument("--steps",      type=int, default=None,  help="Override step budget")
    parser.add_argument("--fps",        type=int, default=20,     help="GIF fps (default 20)")
    parser.add_argument("--controller", default=CONTROLLER,
                        choices=["pure_pursuit", "stanley"], help="Low-level controller")
    parser.add_argument(
        "--map", default=MAP_SEQUENCE,
        help=(
            "Map preset name from my_env.MAP_PRESETS, or a raw block string.\n"
            f"Named presets: {', '.join(_preset_choices)}\n"
            f"Default: {MAP_SEQUENCE!r} (from config.py MAP_SEQUENCE)"
        ),
    )
    parser.add_argument("--lanes",    type=int,   default=3,             help="Number of lanes (default 3)")
    parser.add_argument("--traffic",  type=float, default=TRAFFIC_DENSITY,
                        help=f"Traffic density 0-1 (default {TRAFFIC_DENSITY} from config.py)")
    parser.add_argument("--seed",     type=int,   default=SCENARIO_SEED,
                        help=f"Scenario seed (default {SCENARIO_SEED} from config.py)")
    parser.add_argument("--random-traffic", action="store_true", default=RANDOM_TRAFFIC,
                        help="Randomise NPC routes each run (default: fixed from config.py)")
    parser.add_argument("--no-gif",   action="store_true",      help="Skip saving GIFs")
    parser.add_argument("--no-log",   action="store_true",      help="Disable telemetry logging")
    parser.add_argument("--run-name", default="vlm",            help="Label for the log folder (default: vlm)")
    args = parser.parse_args()

    # Resolve the map display name
    map_display = MAP_PRESETS.get(args.map, args.map)

    print("=" * 60)
    print("  VLM-Guided Motion Planner")
    print(f"  Map preset  : {args.map}  ->  {map_display}")
    print(f"  Lanes       : {args.lanes}")
    print(f"  Traffic     : {args.traffic}  (random_traffic={'ON' if args.random_traffic else 'OFF - same NPCs every run'})")
    print(f"  Seed        : {args.seed}")
    print(f"  VLM         : {args.vlm}")
    print(f"  Controller  : {args.controller}")
    print(f"  Logging     : {'OFF' if args.no_log else 'ON -> logs/' + args.run_name + '_<timestamp>'}")
    print(f"  Instruction : {HUMAN_INSTRUCTION.strip()}")
    print("=" * 60)

    env = make_vlm_env(
        map_preset=args.map,
        traffic_density=args.traffic,
        num_lanes=args.lanes,
        seed=args.seed,
        random_traffic=args.random_traffic,
    )

    run(
        env,
        out_gif_cam=OUT_GIF_CAM,
        out_gif_bev=OUT_GIF_BEV,
        steps=args.steps,
        fps=args.fps,
        instruction=HUMAN_INSTRUCTION,
        bev_size=(800, 800),
        use_vlm=args.vlm,
        controller=args.controller,
        save_gifs=not args.no_gif,
        log=not args.no_log,
        run_name=args.run_name,
    )
