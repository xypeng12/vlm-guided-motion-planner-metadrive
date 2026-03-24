from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_GIF_DIR   = _REPO_ROOT / "save_gif"

OUT_GIF_CAM = str(_GIF_DIR / "metadrive_cam_prompt.gif")
OUT_GIF_BEV = str(_GIF_DIR / "metadrive_bev_cand_prompt.gif")

# ── Map ─────────────────────────────────────────────────────────────────────
# "SSrSRSSrS"  –  simple highway with 2 on-ramps (r) and 1 off-ramp (R)
#   S = Straight   r = InRamp   R = OutRamp
# MAP_SEQUENCE = "SSrSRSrS"
MAP_SEQUENCE = "SSrSSSRS"

# ── Scenario reproducibility ─────────────────────────────────────────────────
# Fixed seed → same NPC spawn positions and routes every run.
# Set RANDOM_TRAFFIC = False to make NPC behavior fully deterministic;
# True lets NPCs choose random routes even with the same seed.
SCENARIO_SEED   = 100
RANDOM_TRAFFIC  = False

# ── Traffic ──────────────────────────────────────────────────────────────────
# 0.0 = no NPC cars, 0.05 = light, 0.15 = moderate, 0.5 = heavy
TRAFFIC_DENSITY = 0.082

# ── Planner timing ───────────────────────────────────────────────────────────
PLANNER_HZ = 10.0
DT = 1.0 / PLANNER_HZ

# Replan every N simulation steps.  Between replans the controller keeps
# following the last chosen trajectory, preventing frame-to-frame oscillation.
# 5 steps @ 10 Hz = replan every 0.5 s.
REPLAN_EVERY_STEPS = 5

# ── Policy switches ──────────────────────────────────────────────────────────
USE_VLM = False         # set True to enable Qwen2-VL; needs GPU / MPS + transformers
# CONTROLLER = "pure_pursuit"   # "pure_pursuit" | "stanley"
CONTROLLER = "stanley"   # "pure_pursuit" | "stanley"

VLM_UPDATE_HZ      = 3.0
VLM_MODEL          = "Qwen/Qwen2-VL-2B-Instruct"
VLM_IMAGE_SIZE     = (512,384)   # smaller → ~2x faster vision encoder; was (512,384)
VLM_USE_IMAGE      = True
VLM_MAX_NEW_TOKENS = 180          # JSON output is ~100-120 tokens; was 200

# ── GIF capture rate ────────────────────────────────────────────────────────
# Capture one GIF frame every N planner steps. At PLANNER_HZ=20:
#   1 → 20 fps GIF (large files, ~3 GB RAM for 90s run)
#   2 → 10 fps GIF (half memory, half CPU for frame building)  <-- default
#   4 →  5 fps GIF (very small files, minimal memory)
GIF_CAPTURE_EVERY = 2

# ── GIF overlay typography ───────────────────────────────────────────────────
GIF_OVERLAY_FONT_SIZE    = 35
GIF_OVERLAY_PANEL_WIDTH  = 440
GIF_OVERLAY_PANEL_HEIGHT = 220
GIF_OVERLAY_MARGIN       = 16
GIF_CAND_LABEL_FONT_SIZE = 28
GIF_CAND_LEGEND_FONT_SIZE = 24

# ── Human instruction passed into the policy ─────────────────────────────────
# Pick ONE instruction to assign to HUMAN_INSTRUCTION.
#
# SAFE – prioritise safety and comfort; stay in lane unless absolutely
#        necessary; keep generous following distance; smooth steering.
HUMAN_INSTRUCTION_SAFE = (
    "Drive safely and comfortably. Stay in the current lane whenever possible. "
    "Only change lanes if the current lane is clearly blocked and the adjacent "
    "lane is safe. Prefer braking over risky lane changes. Prioritise smooth "
    "steering and large clearance from other vehicles over forward speed."
)
#
# AGGRESSIVE / FAST – maximise forward progress; overtake slower traffic by
#                     changing lanes frequently; accept sharper steering.
HUMAN_INSTRUCTION_FAST = (
    "Drive as fast as possible. Actively seek lane changes to overtake slower "
    "traffic. Prioritise forward speed and progress above comfort. Accept "
    "sharper steering and closer proximity to other vehicles, but avoid "
    "direct collisions."
)

# ── Active instruction (swap between _SAFE and _FAST for comparison) ─────────
HUMAN_INSTRUCTION = HUMAN_INSTRUCTION_SAFE

# ── Collision-avoidance parameters ───────────────────────────────────────────
# BLOCK_DETECT_DIST_M: how far ahead (m) a lidar hit triggers "block ahead".
#   18 m ≈ <1 s at highway speed → crashes.  30 m gives ~1.5 s at 20 m/s.
BLOCK_DETECT_DIST_M  = 30.0

# MIN_CLEARANCE_M: hard-feasibility rejection threshold for KeepLane.
#   1.8 m = one car-width.  Larger values cause false rejections from
#   adjacent-lane traffic on a 3.5 m wide highway lane.
MIN_CLEARANCE_M      = 1.8

# FRONT_RANGE_M: obstacle look-ahead used in both block detection (fallback)
#   and hard-feasibility trajectory check.  35 m → 50 m for highway safety.
FRONT_RANGE_M        = 50.0

# LIDAR_MAX_RANGE_M: maximum range of the 240-ray lidar sensor.
#   Must match vehicle_config → lidar → distance in my_env.py.
#   MetaDrive normalises raw readings to [0, 1]; multiply by this to get metres.
LIDAR_MAX_RANGE_M    = 50.0

# ── Objective weights (used when USE_VLM=False) ──────────────────────────────
# Must stay in [0.1, 3.0] — same scale as VLM prompt output — so the two modes
# are directly comparable.  Crash avoidance is handled by the planner
# infrastructure (BLOCK_DETECT_DIST_M, MIN_CLEARANCE_M, FRONT_RANGE_M), NOT by
# inflating weights.
#
#   w_efficiency: how much the planner rewards forward progress (lower = more cautious)
#   w_comfort:    penalty on trajectory curvature (higher = smoother lane changes)
#   w_safety:     reward on clearance to obstacles (higher = more stand-off distance)
DEFAULT_OBJECTIVE_WEIGHTS = {
    "w_efficiency": 1.0,   # slightly below neutral — don't prioritize speed
    "w_comfort":    1.0,   # neutral smoothness
    "w_safety":     1.0,   # prioritise clearance, but within VLM's 0-2 range
}
SIMPLE_POLICY_WEIGHTS = dict(DEFAULT_OBJECTIVE_WEIGHTS)

# ── Lane-keeping preference ───────────────────────────────────────────────────
# Flat score bonus added to KeepLane every planning step.
# Prevents unnecessary lane changes when clearance in the adjacent lane is only
# marginally better.  VLM can still override by outputting bias["ChangeLaneLeft"]
# or bias["ChangeLaneRight"] above this value (both modes use the same bias scale).
KEEP_LANE_BIAS = 1.5

# ── Maneuver set ─────────────────────────────────────────────────────────────
MANEUVERS = ["KeepLane", "ChangeLaneLeft", "ChangeLaneRight", "Brake"]

# ── Curve generation ─────────────────────────────────────────────────────────
CURVE_GENERATION_METHOD = "b_spline"   # "b_spline" | "clothoid"
# CURVE_GENERATION_METHOD = "clothoid"   # "b_spline" | "clothoid"

# ── Debug switches ───────────────────────────────────────────────────────────
DEBUG_DISABLE_FEASIBILITY = False
DEBUG_CANDIDATE_LOG       = False

PRINT_EVERY = 100
