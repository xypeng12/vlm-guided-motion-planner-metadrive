"""
MetaDrive 0.4.3: 3 scenario families + lane-forward-10m target + pure-pursuit controller + save GIF.

Requirements:
pip install metadrive imageio numpy
"""

import os
from typing import Any, Dict, List

import imageio
import numpy as np
from metadrive.envs import MetaDriveEnv
from env import get_family_config, step_compat
from planner import pure_pursuit_to_lane_target, lane_forward_target
from utils import to_uint8


def run_and_save_gif(
    env_cfg: Dict[str, Any],
    out_gif: str,
    steps: int = 500,
    fps: int = 20,
    warmup_steps: int = 12,
    forward_m: float = 10.0,
    v_ref: float = 20.0,
    steer_scale: float = 2.0,   # IMPORTANT: increase if you see understeer
) -> str:
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)

    env = MetaDriveEnv(dict(env_cfg))
    env.reset()
    warmup_render(env, n=warmup_steps)

    cam = env.engine.get_sensor("main_camera")
    frames: List[np.ndarray] = []

    for t in range(steps):
        action = pure_pursuit_to_lane_target(
            env,
            forward_m=forward_m,
            v_ref=v_ref,
            steer_scale=steer_scale,
        )

        if t % 20 == 0:
            tgt = lane_forward_target(env, forward_m=forward_m)
            print(
                f"t={t:04d} action=[steer={action[0]:+.3f}, u={action[1]:+.3f}] "
                f"speed={getattr(env.vehicle,'speed',0.0):.2f} "
                f"tgt={'None' if tgt is None else tgt.tolist()}"
            )

        obs, reward, done, info = step_compat(env, action)

        frame = to_uint8(cam.perceive())
        if frame is not None:
            frames.append(frame)

        if done:
            env.reset()
            warmup_render(env, n=warmup_steps)
            cam = env.engine.get_sensor("main_camera")

    env.close()

    if not frames:
        raise RuntimeError("No frames captured from main_camera. Check use_render/show_logo/warmup_steps.")

    imageio.mimsave(out_gif, frames, fps=fps)
    print(f"Saved: {out_gif} (frames={len(frames)}, fps={fps})")
    return out_gif

def warmup_render(env: MetaDriveEnv, n: int = 12) -> None:
    for _ in range(n):
        env.engine.taskMgr.step()

if __name__ == "__main__":
    family = "blocked"  # baseline | blocked | interactive
    cfg = get_family_config(family)

    run_and_save_gif(
        cfg,
        out_gif=f"sim_env/metadrive_{family}_lane10m_purepursuit.gif",
        steps=500,
        fps=20,
        warmup_steps=15,
        forward_m=10.0,
        v_ref=22.0,
        steer_scale=2.5,   # if still not turning enough, try 3.5 or 4.0
    )