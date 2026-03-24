from utils import to_uint8, pick_device
import json
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
except Exception as _vlm_import_err:
    torch = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModelForVision2Seq = None  # type: ignore

from config import VLM_UPDATE_HZ, VLM_IMAGE_SIZE, VLM_MODEL, VLM_USE_IMAGE, MANEUVERS, VLM_MAX_NEW_TOKENS, DEFAULT_OBJECTIVE_WEIGHTS

# =========================
# VLM Policy
# =========================
class Qwen2VLPolicy:
    def __init__(self, model_name: str = VLM_MODEL, max_new_tokens: int = VLM_MAX_NEW_TOKENS):
        self.device = pick_device()
        self.processor = AutoProcessor.from_pretrained(model_name)

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device
        )
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.last = {
            "bias": {m: 0.0 for m in MANEUVERS},
            "weights": dict(DEFAULT_OBJECTIVE_WEIGHTS),
            "notes": "init",
            "_prompt": "",
            "_raw": "",
        }

    def _prompt(self, instruction: str, summary: dict) -> str:
        ex_aggressive = (
            "Example \u2014 aggressive instruction:\n"
            '{"bias":{"KeepLane":-1.0,"ChangeLaneLeft":2.0,"ChangeLaneRight":2.0,"Brake":-2.0},'
            '"weights":{"w_efficiency":3.0,"w_comfort":0.1,"w_safety":0.1},'
            '"notes":"Maximum aggression."}\n\n'
        )
        ex_safe = (
            "Example \u2014 safe/comfort instruction:\n"
            '{"bias":{"KeepLane":2.0,"ChangeLaneLeft":-2.0,"ChangeLaneRight":-2.0,"Brake":1.0},'
            '"weights":{"w_efficiency":0.5,"w_comfort":3.0,"w_safety":3.0},'
            '"notes":"Safety and comfort first."}\n\n'
        )
        # Show only the MATCHING example — no competing pattern for the 2B
        # model to anchor to.
        inst_lower = instruction.lower()
        is_aggressive = any(w in inst_lower for w in ("fast", "speed", "aggress", "overtake"))
        if is_aggressive:
            examples = ex_aggressive
        else:
            examples = ex_safe

        return (
            "You are a driving preference advisor. Given a driving instruction and road state, "
            "output a single JSON object with bias and weights matching the style shown in the example.\n\n"
            "Rules:\n"
            "- bias: positive=encourage, negative=discourage\n"
            "- KeepLane and ChangeLane biases must have OPPOSITE signs\n"
            "- Use values SIMILAR to the example below\n"
            "- Output ONLY a JSON object, no extra text\n\n"
            + examples
            + f"Road state: {json.dumps(summary, ensure_ascii=False)}\n\n"
            f"Instruction: {instruction}\n"
            "JSON:"
        )

    @staticmethod
    def _safe_parse(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from VLM output, handling markdown code blocks and extra text."""
        if not isinstance(text, str) or len(text) == 0:
            return None
        
        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # Strip markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Try again after stripping markdown
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # Find JSON object boundaries
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

        # Fresh defaults for this call — never carry over previous weights as fallback
        _defaults: Dict[str, Any] = {
            "bias": {m: 0.0 for m in MANEUVERS},
            "weights": dict(DEFAULT_OBJECTIVE_WEIGHTS),
            "notes": "",
            "_prompt": prompt,
            "_raw": "",
        }

        def _fallback(note: str, raw: str = "") -> Dict[str, Any]:
            result = dict(_defaults)
            result["_raw"] = raw
            result["notes"] = note
            self.last = result
            return result

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                )
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            out_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            parsed = self._safe_parse(out_text)
            if parsed is None or not isinstance(parsed, dict):
                print(f"[VLM] Parse failed. Raw output:\n{out_text}")
                return _fallback("parse_failed_fallback", out_text)

            bias = parsed.get("bias", None)
            weights = parsed.get("weights", None)
            if not isinstance(bias, dict) or not isinstance(weights, dict):
                print(f"[VLM] Schema failed. Parsed: {parsed}")
                print(f"  bias type: {type(bias)}, weights type: {type(weights)}")
                return _fallback("schema_failed_fallback", out_text)

            def clamp(x, lo=-3.0, hi=3.0, default=0.0):
                try:
                    return float(np.clip(float(x), lo, hi))
                except Exception:
                    return float(default)

            def clamp_w(x, lo=0.1, hi=3.0, default=1.0):
                try:
                    return float(np.clip(float(x), lo, hi))
                except Exception:
                    return float(default)

            ret = {
                "bias": {m: clamp(bias.get(m, 0.0)) for m in MANEUVERS},
                "weights": {
                    "w_efficiency": clamp_w(weights.get("w_efficiency", weights.get("w_progress", DEFAULT_OBJECTIVE_WEIGHTS["w_efficiency"]))),
                    "w_comfort": clamp_w(weights.get("w_comfort", DEFAULT_OBJECTIVE_WEIGHTS["w_comfort"])),
                    "w_safety": clamp_w(weights.get("w_safety", weights.get("w_clearance", DEFAULT_OBJECTIVE_WEIGHTS["w_safety"]))),
                },
                "notes": str(parsed.get("notes", ""))[:200],
                "_prompt": prompt,
                "_raw": out_text,
            }

            # ── Post-process: enforce bias sign consistency ──────────────
            # KeepLane and ChangeLane biases should have opposite signs.
            # If they match, flip ChangeLane to be opposite of KeepLane.
            b_kl = ret["bias"].get("KeepLane", 0.0)
            b_cl = ret["bias"].get("ChangeLaneLeft", 0.0)
            b_cr = ret["bias"].get("ChangeLaneRight", 0.0)
            if b_kl > 0 and (b_cl > 0 or b_cr > 0):
                ret["bias"]["ChangeLaneLeft"]  = -abs(b_cl)
                ret["bias"]["ChangeLaneRight"] = -abs(b_cr)
            elif b_kl < 0 and (b_cl < 0 or b_cr < 0):
                ret["bias"]["ChangeLaneLeft"]  = abs(b_cl)
                ret["bias"]["ChangeLaneRight"] = abs(b_cr)

            _bias_str = " ".join(f"{k}={v:+.1f}" for k, v in ret['bias'].items())
            print(f"[VLM] eff={ret['weights']['w_efficiency']:.2f} com={ret['weights']['w_comfort']:.2f} saf={ret['weights']['w_safety']:.2f} | bias: {_bias_str} | {ret['notes'][:60]}")
            self.last = ret
            return ret

        except Exception as e:
            return _fallback(f"exception_fallback: {type(e).__name__}: {e}", raw="")
