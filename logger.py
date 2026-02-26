"""
logger.py  –  Per-run telemetry logger for MAE252 MetaDrive
============================================================
Writes per run into  logs/<timestamp>_<name>/:
  telemetry.csv  – full time-series data every step
  summary.csv    – single-row run statistics
  run.mp4        – 3-D window video (when record_video=True)
"""

import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


LOG_ROOT = Path(__file__).parent / "logs"


class RunLogger:
    """Attach to a MetaDrive env and call .record() each step."""

    COLUMNS = [
        "timestep", "elapsed_s",
        "x", "y", "heading_deg",
        "speed_ms", "speed_kmh",
        "vx", "vy",
        "accel_ms2",
        "steering", "throttle_brake",
        "total_dist_m", "lane_changes",
        "on_broken_line", "crash_vehicle", "crash_object",
        "reward", "cost",
        "current_lane_id",
    ]

    def __init__(self, run_name: str = "", record_video: bool = False,
                 video_fps: int = 30, video_size: int = 800):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{ts}_{run_name}" if run_name else ts
        self.run_dir = LOG_ROOT / label
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "telemetry.csv"
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()

        # state tracking
        self._start_time   = time.perf_counter()
        self._prev_speed   = 0.0
        self._prev_pos     = None
        self._prev_lane_id = None
        self._total_dist   = 0.0
        self._lane_changes = 0

        # video
        self._video_writer = None
        self._video_path   = None
        self._video_fps    = video_fps
        self._video_size   = video_size  # (w, h) matched to window_size config
        self._record_video = record_video
        self._frame_errors = 0
        self._last_frame_time = 0.0   # wall-clock time of last written frame

        if record_video:
            self._video_path = self.run_dir / "run.mp4"
            # VideoWriter is opened on the first frame once we know the real size.
            print(f"[Logger] Video will be recorded → {self._video_path}")

        print(f"[Logger] Writing telemetry → {self.csv_path}")

    # ── main record call ───────────────────────────────────────────────────────
    def record(self, env, reward: float = 0.0, info: dict = None):
        """Call once per env.step(). Pass the env object.

        If record_video=True, automatically captures a top-down frame
        (does not affect the 3-D window).
        """
        info = info or {}
        agent = env.agent
        step  = env.engine.episode_step

        # position & heading
        pos = agent.position          # numpy (x, y)  or panda3d Point3
        x, y = float(pos[0]), float(pos[1])
        heading_rad = float(agent.heading_theta)
        heading_deg = math.degrees(heading_rad)

        # speed & velocity
        speed_ms  = float(agent.speed)          # m/s
        speed_kmh = float(agent.speed_km_h)

        vel = agent.velocity                    # (vx, vy)
        vx, vy = float(vel[0]), float(vel[1])

        # acceleration (finite-difference on speed)
        dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
        accel = (speed_ms - self._prev_speed) / dt if dt > 0 else 0.0
        self._prev_speed = speed_ms

        # controls
        steering      = float(agent.steering)
        throttle_brake = float(agent.throttle_brake)

        # cumulative distance
        if self._prev_pos is not None:
            dx = x - self._prev_pos[0]
            dy = y - self._prev_pos[1]
            self._total_dist += math.sqrt(dx * dx + dy * dy)
        self._prev_pos = (x, y)

        # lane change count
        try:
            lane_id = str(agent.lane_index)
        except Exception:
            lane_id = ""
        if self._prev_lane_id is not None and lane_id != self._prev_lane_id:
            self._lane_changes += 1
        self._prev_lane_id = lane_id

        # safety flags
        on_broken   = int(bool(getattr(agent, "on_broken_line", False)))
        crash_veh   = int(bool(agent.crash_vehicle))
        crash_obj   = int(bool(agent.crash_object))

        cost = float(info.get("cost", 0.0))
        elapsed = time.perf_counter() - self._start_time

        row = {
            "timestep":      step,
            "elapsed_s":     round(elapsed, 4),
            "x":             round(x, 4),
            "y":             round(y, 4),
            "heading_deg":   round(heading_deg, 3),
            "speed_ms":      round(speed_ms, 4),
            "speed_kmh":     round(speed_kmh, 4),
            "vx":            round(vx, 4),
            "vy":            round(vy, 4),
            "accel_ms2":     round(accel, 5),
            "steering":      round(steering, 5),
            "throttle_brake": round(throttle_brake, 5),
            "total_dist_m":  round(self._total_dist, 3),
            "lane_changes":  self._lane_changes,
            "on_broken_line": on_broken,
            "crash_vehicle": crash_veh,
            "crash_object":  crash_obj,
            "reward":        round(reward, 5),
            "cost":          cost,
            "current_lane_id": lane_id,
        }
        self._writer.writerow(row)

        # ── video frame capture (3-D window) ────────────────────────────────────
        if self._record_video:
            now = time.perf_counter()
            frame_interval = 1.0 / self._video_fps
            # Only capture a frame when enough real time has elapsed.
            # This makes the video play back at real speed even if the
            # simulation runs faster than video_fps.
            if now - self._last_frame_time < frame_interval:
                return
            self._last_frame_time = now
            try:
                # env.engine._get_window_image() screenshots the panda3D
                # render window and returns an RGB (H, W, 3) uint8 array.
                frame = env.engine._get_window_image()

                # Lazily open the VideoWriter on the first real frame so the
                # dimensions are guaranteed to match the actual window.
                if self._video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._video_writer = cv2.VideoWriter(
                        str(self._video_path), fourcc, self._video_fps, (w, h)
                    )
                    if not self._video_writer.isOpened():
                        # Fall back to XVID + AVI
                        self._video_path = self._video_path.with_suffix(".avi")
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        self._video_writer = cv2.VideoWriter(
                            str(self._video_path), fourcc, self._video_fps, (w, h)
                        )
                    if self._video_writer.isOpened():
                        print(f"[Logger] VideoWriter opened → {self._video_path} ({w}x{h} @{self._video_fps}fps)")
                    else:
                        print("[Logger] WARNING: VideoWriter failed to open – video disabled")
                        self._video_writer = None
                        self._record_video = False

                if self._video_writer is not None:
                    # panda3D gives RGB; OpenCV needs BGR
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._video_writer.write(bgr)

            except Exception as exc:
                self._frame_errors += 1
                if self._frame_errors <= 5:
                    print(f"[Logger] Frame capture error #{self._frame_errors}: {exc}")

    def close(self, summary: bool = True):
        self._file.flush()
        self._file.close()
        if self._video_writer is not None:
            self._video_writer.release()
            print(f"[Logger] Video saved  → {self._video_path}")
        if summary:
            self._write_summary()
        print(f"[Logger] Run finished. Rows written to {self.csv_path}")

    # ── end-of-run summary ────────────────────────────────────────────────────
    def _write_summary(self):
        elapsed = time.perf_counter() - self._start_time
        total_steps = self._get_last_step()

        summary = {
            "run_dir":        str(self.run_dir.name),
            "total_steps":    total_steps,
            "elapsed_s":      round(elapsed, 2),
            "total_dist_m":   round(self._total_dist, 2),
            "lane_changes":   self._lane_changes,
        }

        # ── derive stats from the telemetry CSV ──────────────────────────────
        try:
            import csv as _csv
            speeds, accels, costs = [], [], []
            crashes_veh = crashes_obj = broken_line_steps = 0
            with open(self.csv_path, newline="") as f:
                for row in _csv.DictReader(f):
                    speeds.append(float(row["speed_kmh"]))
                    accels.append(float(row["accel_ms2"]))
                    costs.append(float(row["cost"]))
                    crashes_veh    += int(row["crash_vehicle"])
                    crashes_obj    += int(row["crash_object"])
                    broken_line_steps += int(row["on_broken_line"])

            def _safe(fn, lst): return round(fn(lst), 4) if lst else 0.0

            summary["avg_speed_kmh"]    = _safe(lambda x: sum(x)/len(x), speeds)
            summary["max_speed_kmh"]    = _safe(max, speeds)
            summary["avg_accel_ms2"]    = _safe(lambda x: sum(x)/len(x), accels)
            summary["max_accel_ms2"]    = _safe(max, accels)
            summary["min_accel_ms2"]    = _safe(min, accels)
            summary["total_cost"]       = round(sum(costs), 4)
            summary["crash_vehicle_steps"]   = crashes_veh
            summary["crash_object_steps"]    = crashes_obj
            summary["broken_line_steps"] = broken_line_steps
        except Exception as e:
            summary["stat_error"] = str(e)

        # ── write CSV ─────────────────────────────────────────────────────────
        summary_csv = self.run_dir / "summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        print(f"[Logger] Summary  → {summary_csv}")

    def _get_last_step(self):
        # count rows written (minus header)
        try:
            with open(self.csv_path) as f:
                return sum(1 for _ in f) - 1
        except Exception:
            return -1
