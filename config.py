from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_GIF_DIR = _REPO_ROOT / "save_gif"

OUT_GIF_CAM = str(_GIF_DIR / "metadrive_cam_prompt.gif")
OUT_GIF_BEV = str(_GIF_DIR / "metadrive_bev_cand_prompt.gif")

PLANNER_HZ = 10.0
DT = 1.0 / PLANNER_HZ

USE_VLM = True   #  1!. False and True
CONTROLLER = "pure_puresuit" # "stanley"  #2!. pure_puresuit, stanley

VLM_UPDATE_HZ = 1.0
VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct" # 3?. Qwen/Qwen2-VL-2B-Instruct, llava-hf/llava-v1.6-mistral-7b-hf, Qwen/Qwen2-VL-7B-Instruct
# llava-hf/llava-v1.6-mistral-7b-hf
# Qwen/Qwen2-VL-7B-Instruct

VLM_IMAGE_SIZE = (512, 384)
VLM_USE_IMAGE = True

# GIF overlay typography
GIF_OVERLAY_FONT_SIZE = 35
GIF_OVERLAY_PANEL_WIDTH = 440
GIF_OVERLAY_PANEL_HEIGHT = 220
GIF_OVERLAY_MARGIN = 16
GIF_CAND_LABEL_FONT_SIZE = 28
GIF_CAND_LEGEND_FONT_SIZE = 24

# Scenario: "baseline" | "blocked" | "interactive"
SCENARIO_FAMILY = "blocked"  # 4. "baseline" | "blocked" | "interactive"

# User-level instruction passed into the policy prompt.
HUMAN_INSTRUCTION = (
    "Drive safely. If there is a block ahead, change lane smoothly if safe; otherwise slow down."
)  # 5. instruction

# Three-aspect objective defaults, without VLM
DEFAULT_OBJECTIVE_WEIGHTS = {
    "w_efficiency": 0.9,
    "w_comfort": 0.7,
    "w_safety": 1.2,
}  # 6. weights

# Used by SimpleTextHeuristicPolicy when USE_VLM=False
SIMPLE_POLICY_WEIGHTS = dict(DEFAULT_OBJECTIVE_WEIGHTS)

# Curve generation for trajectory candidates: "b_spline" | "clothoid"
CURVE_GENERATION_METHOD = "b_spline" # 7!. "b_spline", "clothoid"

MANEUVERS = ["KeepLane", "ChangeLaneLeft", "ChangeLaneRight", "Brake"]
PRINT_EVERY = 20

# Debug switch: bypass hard feasibility rejection to inspect raw candidate generation/scoring.
DEBUG_DISABLE_FEASIBILITY = False
DEBUG_CANDIDATE_LOG = False