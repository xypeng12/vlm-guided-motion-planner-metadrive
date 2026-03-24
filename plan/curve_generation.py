from typing import Literal

import numpy as np


CurveMethod = Literal["b_spline", "clothoid"]


def _resample_polyline(ctrl_pts: np.ndarray, n_samples: int = 80) -> np.ndarray:
    pts = np.asarray(ctrl_pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    if len(pts) == 1:
        return np.repeat(pts, n_samples, axis=0)

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], n_samples, axis=0)

    q = np.linspace(0.0, total, n_samples, dtype=np.float32)
    out = np.zeros((n_samples, 2), dtype=np.float32)
    j = 0
    for i, qi in enumerate(q):
        while (j + 1) < len(s) and float(s[j + 1]) < float(qi):
            j += 1
        if (j + 1) >= len(s):
            out[i] = pts[-1]
            continue
        ds = float(s[j + 1] - s[j])
        if ds < 1e-6:
            out[i] = pts[j]
            continue
        u = float((qi - s[j]) / ds)
        out[i] = (1.0 - u) * pts[j] + u * pts[j + 1]
    return out


def bspline_curve(control_pts: np.ndarray, n_samples: int = 70, degree: int = 3) -> np.ndarray:
    """
    Open-uniform B-spline evaluation in 2D.
    control_pts: [M,2]
    returns: [n_samples,2]
    """
    p = np.asarray(control_pts, dtype=np.float32)
    m = p.shape[0]
    k = degree
    if m == 0:
        return p
    if m < k + 1:
        idx = np.linspace(0, m - 1, n_samples).astype(int)
        return p[idx]

    n_knots = m + k + 1
    knots = np.zeros(n_knots, dtype=np.float32)
    knots[k : m + 1] = np.linspace(0.0, 1.0, m - k + 1)
    knots[m + 1 :] = 1.0

    def basis(i: int, kk: int, t: float) -> float:
        if kk == 0:
            return 1.0 if (knots[i] <= t < knots[i + 1] or (t == 1.0 and knots[i + 1] == 1.0)) else 0.0
        denom1 = knots[i + kk] - knots[i]
        denom2 = knots[i + kk + 1] - knots[i + 1]
        term1 = 0.0
        term2 = 0.0
        if denom1 > 1e-8:
            term1 = (t - knots[i]) / denom1 * basis(i, kk - 1, t)
        if denom2 > 1e-8:
            term2 = (knots[i + kk + 1] - t) / denom2 * basis(i + 1, kk - 1, t)
        return term1 + term2

    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    c = np.zeros((n_samples, 2), dtype=np.float32)
    for ti, t in enumerate(ts):
        pt = np.zeros(2, dtype=np.float32)
        wsum = 0.0
        for i in range(m):
            b = basis(i, k, float(t))
            wsum += b
            pt += b * p[i]
        if wsum > 1e-8:
            pt /= wsum
        c[ti] = pt
    return c


def clothoid_curve(control_pts: np.ndarray, n_samples: int = 80) -> np.ndarray:
    """
    Clothoid-like generator from control polyline:
    - linear-curvature fit k(s)=a*s+b
    - integrates heading/position
    - endpoint correction and mild blend with base polyline
    """
    base = _resample_polyline(control_pts, n_samples=n_samples)
    if len(base) < 3:
        return base

    seg = base[1:] - base[:-1]
    ds = np.linalg.norm(seg, axis=1)
    ds = np.maximum(ds, 1e-6)
    s_nodes = np.concatenate([[0.0], np.cumsum(ds)])
    l = float(s_nodes[-1])
    if l < 1e-6:
        return base

    theta_seg = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    theta_nodes = np.concatenate([[theta_seg[0]], theta_seg]).astype(np.float32)
    k_nodes = np.gradient(theta_nodes, s_nodes, edge_order=1)

    a_mat = np.stack([s_nodes, np.ones_like(s_nodes)], axis=1)
    ab, _, _, _ = np.linalg.lstsq(a_mat, k_nodes, rcond=None)
    a = float(ab[0])
    b = float(ab[1])

    theta0 = float(theta_nodes[0])
    theta = theta0 + b * s_nodes + 0.5 * a * s_nodes * s_nodes

    # End-heading correction while preserving clothoid-like smoothness.
    d_end = float((theta_nodes[-1] - theta[-1] + np.pi) % (2.0 * np.pi) - np.pi)
    theta = theta + (s_nodes / l) * d_end

    out = np.zeros_like(base)
    out[0] = base[0]
    for i in range(1, len(out)):
        dsi = float(s_nodes[i] - s_nodes[i - 1])
        thm = 0.5 * float(theta[i] + theta[i - 1])
        out[i, 0] = out[i - 1, 0] + dsi * np.cos(thm)
        out[i, 1] = out[i - 1, 1] + dsi * np.sin(thm)

    # Force exact endpoint; distribute correction smoothly along arc length.
    end_err = base[-1] - out[-1]
    corr = (s_nodes / l)[:, None] * end_err[None, :]
    out = out + corr

    # Blend with base polyline for robustness on short/noisy sections.
    out = 0.75 * out + 0.25 * base
    out[0] = base[0]
    out[-1] = base[-1]
    return out.astype(np.float32)


def generate_curve(
    control_pts: np.ndarray,
    method: CurveMethod = "b_spline",
    n_samples: int = 80,
    degree: int = 3,
) -> np.ndarray:
    m = str(method).lower().strip()
    if m == "clothoid":
        return clothoid_curve(control_pts, n_samples=n_samples)
    # default/fallback
    return bspline_curve(control_pts, n_samples=n_samples, degree=degree)
