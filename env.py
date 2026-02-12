from typing import Any, Dict, List

from metadrive.envs import MetaDriveEnv

# -------------------------
# 1) Scenario families (30 scenarios each), random traffic ON
# -------------------------
COMMON: Dict[str, Any] = dict(
    use_render=True,          # needed for main_camera
    show_logo=False,          # helps avoid capturing splash
    show_interface=False,     # lighter on macOS
    horizon=1000,
    out_of_route_done=True,
    random_traffic=True,
    random_spawn_lane_index=True,
    num_agents=1,
    map="r",
)

SCENE_BASELINE: Dict[str, Any] = dict(
    **COMMON,
    start_seed=100,
    num_scenarios=30,
    traffic_density=0.05,
    need_inverse_traffic=False,
    static_traffic_object=False,
    accident_prob=0.0,
    crash_vehicle_done=True,
)

SCENE_BLOCKED: Dict[str, Any] = dict(
    **COMMON,
    start_seed=200,
    num_scenarios=30,
    traffic_density=0.08,
    need_inverse_traffic=False,
    static_traffic_object=True,
    accident_prob=0.6,
    crash_vehicle_done=True,
)

SCENE_INTERACTIVE: Dict[str, Any] = dict(
    **COMMON,
    start_seed=300,
    num_scenarios=30,
    traffic_density=0.12,
    need_inverse_traffic=True,
    static_traffic_object=False,
    accident_prob=0.0,
    crash_vehicle_done=True,
)


def get_family_config(family: str) -> Dict[str, Any]:
    fam = family.lower().strip()
    if fam in ["baseline", "base"]:
        return dict(SCENE_BASELINE)
    if fam in ["blocked", "block", "stalled"]:
        return dict(SCENE_BLOCKED)
    if fam in ["interactive", "interact", "hard"]:
        return dict(SCENE_INTERACTIVE)
    raise ValueError(f"Unknown family: {family}")

def step_compat(env: MetaDriveEnv, action: List[float]):
    ret = env.step(action)
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = ret
    return obs, reward, bool(done), info
