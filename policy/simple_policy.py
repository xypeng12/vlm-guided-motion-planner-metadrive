from typing import Any, Dict, Optional
import numpy as np

from config import MANEUVERS, SIMPLE_POLICY_WEIGHTS

class SimpleTextHeuristicPolicy:
    def __init__(self):
        self.last = {
            "bias": {m: 0.0 for m in MANEUVERS},
            "weights": dict(SIMPLE_POLICY_WEIGHTS),
            "notes": "heuristic default",
            "_prompt": "",
            "_raw": "",
        }

    def __call__(self, instruction: str, summary: dict, frame_rgb: Optional[np.ndarray]) -> Dict[str, Any]:
        return self.last
