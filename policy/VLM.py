from utils import to_uint8, pick_device
import json
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

VLM_UPDATE_HZ = 1.0
VLM_IMAGE_SIZE = (512, 384)
VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
VLM_USE_IMAGE = True
MANEUVERS = ["KeepLane", "ChangeLaneLeft", "ChangeLaneRight", "Brake"]

# =========================
# VLM Policy
# =========================
class Qwen2VLPolicy:
    def __init__(self, model_name: str = VLM_MODEL, max_new_tokens: int = 200):
        self.device = pick_device()
        self.processor = AutoProcessor.from_pretrained(model_name)

        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.last = {
            "bias": {m: 0.0 for m in MANEUVERS},
            "weights": {
                "w_efficiency": 0.9,
                "w_comfort": 0.7,
                "w_safety": 1.2,
            },
            "notes": "init",
            "_prompt": "",
            "_raw": "",
        }

    def _prompt(self, instruction: str, summary: dict) -> str:
        schema = {
            "bias": {m: "-3.0..3.0" for m in MANEUVERS},
            "weights": {
                "w_efficiency": "0.0-2.0",
                "w_comfort": "0.0-2.0",
                "w_safety": "0.0-2.0"
            },
            "notes": "short reason"
        }
        return (
            "You are a driving preference advisor.\n"
            "You do NOT do safety checking and you do NOT output control actions.\n"
            "Only output preferences (bias) and scoring weights.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"State summary (JSON):\n{json.dumps(summary, ensure_ascii=False)}\n\n"
            "Return ONLY valid JSON following this schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            "Rules:\n"
            "- bias MUST contain all 4 maneuvers. If unsure, set 0.\n"
            "- weights MUST reflect instruction priorities among efficiency/comfort/safety.\n"
            "- larger weight means that aspect is more important right now.\n"
            "- weights are floats.\n"
            "- No markdown. No extra words outside JSON.\n"
        )

    @staticmethod
    def _safe_parse(text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str) or len(text) == 0:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        l = text.find("{")
        r = text.rfind("}")
        if l == -1 or r == -1 or r <= l:
            return None
        cand = text[l:r + 1]
        try:
            return json.loads(cand)
        except Exception:
            return None

    def __call__(self, instruction: str, summary: dict, frame_rgb: Optional[np.ndarray]) -> Dict[str, Any]:
        prompt = self._prompt(instruction, summary)

        if frame_rgb is not None:
            img = Image.fromarray(to_uint8(frame_rgb))
            img = img.resize(VLM_IMAGE_SIZE)
            messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        def _fallback(note: str, raw: str = "") -> Dict[str, Any]:
            last = dict(self.last)
            last["_prompt"] = prompt
            last["_raw"] = raw
            last["notes"] = note
            self.last = last
            return last

        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            out_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            parsed = self._safe_parse(out_text)
            if parsed is None or not isinstance(parsed, dict):
                return _fallback("parse_failed_fallback", out_text)

            bias = parsed.get("bias", None)
            weights = parsed.get("weights", None)
            if not isinstance(bias, dict) or not isinstance(weights, dict):
                return _fallback("schema_failed_fallback", out_text)

            def clamp(x, lo=-3.0, hi=3.0, default=0.0):
                try:
                    return float(np.clip(float(x), lo, hi))
                except Exception:
                    return float(default)

            def clamp_w(x, lo=0.0, hi=2.0, default=0.5):
                try:
                    return float(np.clip(float(x), lo, hi))
                except Exception:
                    return float(default)

            ret = {
                "bias": {m: clamp(bias.get(m, 0.0)) for m in MANEUVERS},
                "weights": {
                    "w_efficiency": clamp_w(weights.get("w_efficiency", weights.get("w_progress", 0.9))),
                    "w_comfort": clamp_w(weights.get("w_comfort", 0.7)),
                    "w_safety": clamp_w(weights.get("w_safety", weights.get("w_clearance", 1.2))),
                },
                "notes": str(parsed.get("notes", ""))[:200],
                "_prompt": prompt,
                "_raw": out_text,
            }
            self.last = ret
            return ret

        except Exception as e:
            return _fallback(f"exception_fallback: {type(e).__name__}: {e}", raw="")
