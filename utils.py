from typing import Optional


import numpy as np

def wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def to_uint8(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return None
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    if not np.isfinite(f).all():
        return None
    if float(np.max(f)) <= 1.0 + 1e-6:
        f = f * 255.0
    return np.clip(f, 0.0, 255.0).astype(np.uint8)
