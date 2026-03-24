from PIL import ImageDraw, ImageFont
import textwrap
from utils import to_uint8
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image


def _load_font(font_size: int):
    size = max(10, int(font_size))
    font_candidates = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for fpath in font_candidates:
        try:
            return ImageFont.truetype(fpath, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def _wrap_lines(s: str, width: int = 90, max_lines: int = 10) -> List[str]:
    if s is None:
        return []
    s = str(s)
    lines = []
    for chunk in s.splitlines():
        lines.extend(textwrap.wrap(chunk, width=width) if chunk.strip() else [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return lines


def _fit_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if max_w <= 8 or font is None:
        return s
    try:
        bb = draw.textbbox((0, 0), s, font=font)
        if (bb[2] - bb[0]) <= max_w:
            return s
    except Exception:
        return s
    ell = "..."
    lo, hi = 0, len(s)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = s[:mid] + ell
        try:
            bb = draw.textbbox((0, 0), cand, font=font)
            w = bb[2] - bb[0]
        except Exception:
            return cand
        if w <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def draw_text_panel(
    img_rgb: np.ndarray,
    lines: List[Any],
    panel_w: Optional[int] = None,
    panel_h: Optional[int] = None,
    alpha: int = 140,
    max_chars: int = 220,
    font_size: int = 24,
    x: int = 0,
    y: int = 0,
) -> np.ndarray:
    if img_rgb is None:
        return img_rgb

    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im, "RGBA")
    font = _load_font(font_size)
    col_font = _load_font(max(12, int(font_size * 0.58)))

    pad = max(8, int(font_size * 0.28))
    if font is None:
        line_h = 12
    else:
        try:
            bb = font.getbbox("Ag")
            text_h = int(bb[3] - bb[1])
        except Exception:
            text_h = int(getattr(font, "size", 12))
        line_h = max(12, text_h + max(6, int(font_size * 0.25)))

    if panel_w is None:
        panel_w = int(im.size[0] * 0.65)
    panel_w = int(min(max(1, panel_w), im.size[0]))
    if panel_h is None:
        panel_h = pad * 2 + line_h * max(1, len(lines))
    panel_h = int(min(max(1, panel_h), im.size[1]))

    x = int(max(0, min(int(x), im.size[0] - panel_w)))
    y = int(max(0, min(int(y), im.size[1] - panel_h)))
    draw.rectangle([x, y, x + panel_w, y + panel_h], fill=(96, 96, 96, int(alpha)))

    yy = y + pad
    stroke_w = max(1, int(font_size * 0.08))
    col_stroke_w = max(1, int(font_size * 0.05))
    for ln in lines:
        if isinstance(ln, (list, tuple)):
            cols = [str(t) for t in ln]
            ncol = max(1, len(cols))
            col_w = max(1, int((panel_w - 2 * pad) / ncol))
            line_right = x + panel_w - pad
            for ci, ctext in enumerate(cols):
                xx = x + pad + ci * col_w
                if ci == ncol - 1:
                    usable_w = max(8, line_right - xx)
                else:
                    usable_w = max(8, col_w - int(pad * 0.6))
                text = _fit_text_to_width(
                    draw,
                    ctext[: max(10, int(max_chars / ncol))],
                    col_font or font,
                    usable_w,
                )
                try:
                    draw.text(
                        (xx, yy),
                        text,
                        fill=(255, 255, 255, 255),
                        font=col_font or font,
                        stroke_width=col_stroke_w,
                        stroke_fill=(0, 0, 0, 255),
                    )
                except TypeError:
                    draw.text((xx, yy), text, fill=(255, 255, 255, 255), font=col_font or font)
                if ci < ncol - 1:
                    sep_x = xx + col_w - int(pad * 0.35)
                    draw.line(
                        [(sep_x, yy + 2), (sep_x, yy + line_h - 4)],
                        fill=(200, 200, 200, 180),
                        width=1,
                    )
        else:
            text = str(ln)[:max_chars]
            try:
                draw.text(
                    (x + pad, yy),
                    text,
                    fill=(255, 255, 255, 255),
                    font=font,
                    stroke_width=stroke_w,
                    stroke_fill=(0, 0, 0, 255),
                )
            except TypeError:
                draw.text((x + pad, yy), text, fill=(255, 255, 255, 255), font=font)
        yy += line_h

    return np.asarray(im)


def get_topdown_frame(env, size: Tuple[int, int] = (800, 800)) -> Optional[np.ndarray]:
    """
    Robust topdown capture for MetaDrive 0.4.3-ish.
    Try several signatures until one works.
    """
    # 1) recommended signature in many versions
    try:
        frame = env.render(
            mode="topdown",
            window=False,
            screen_size=size,
            scaling=5,          # px per meter (tune)
        )
        frame = to_uint8(frame)
        if frame is None:
            return None
        return frame[..., :3]
    except TypeError:
        pass
    except Exception:
        pass

    # 2) some versions use film_size
    try:
        frame = env.render(mode="topdown", film_size=size)
        frame = to_uint8(frame)
        if frame is None:
            return None
        return frame[..., :3]
    except TypeError:
        pass
    except Exception:
        pass

    # 3) fallback: no args
    try:
        frame = env.render(mode="topdown")
        frame = to_uint8(frame)
        if frame is None:
            return None
        return frame[..., :3]
    except Exception:
        return None


def world_to_topdown_px(env, pts_xy: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert world (x,y) to topdown pixel coords using env.engine.top_down_renderer.
    This ONLY works AFTER you've called env.render(mode="topdown") at least once in this episode.
    """
    if pts_xy is None or len(pts_xy) == 0:
        return None

    r = getattr(env.engine, "top_down_renderer", None)
    if r is None:
        return None

    pts = np.asarray(pts_xy, dtype=np.float32)
    out = []

    # try common method names across versions
    for p in pts:
        x, y = float(p[0]), float(p[1])
        pix = None

        # most common in MetaDrive is world_to_pixel((x,y))
        for name in ["world_to_pixel", "pos2pix", "position_to_pixel", "vec2pix"]:
            if hasattr(r, name):
                fn = getattr(r, name)
                try:
                    # try tuple arg first
                    pix = fn((x, y))
                    break
                except Exception:
                    try:
                        pix = fn(x, y)
                        break
                    except Exception:
                        pix = None

        if pix is None:
            return None

        out.append([float(pix[0]), float(pix[1])])

    return np.asarray(out, dtype=np.float32)


def draw_polylines_rgb(rgb: np.ndarray, polylines: List[np.ndarray]) -> np.ndarray:
    if rgb is None:
        return rgb
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for pts in polylines:
        if pts is None or len(pts) < 2:
            continue
        xy = [(int(p[0]), int(p[1])) for p in pts]
        draw.line(xy, fill=(255, 255, 255), width=2)
    return np.asarray(img)


def draw_ranked_candidates(
    rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    top_k_highlight: int = 3,
    label_font_size: int = 14,
    legend_font_size: int = 13,
) -> np.ndarray:
    """
    Draw ranked candidate trajectories:
    - all candidates in faint white
    - top-k in distinct colors/thicker lines, with rank label near start
    """
    if rgb is None:
        return rgb

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font_label = _load_font(label_font_size)
    font_legend = _load_font(legend_font_size) or font_label

    # Draw all candidates first (faint context)
    for c in candidates:
        pts = c.get("px", None)
        if pts is None or len(pts) < 2:
            continue
        xy = [(int(p[0]), int(p[1])) for p in pts]
        draw.line(xy, fill=(180, 180, 180), width=2)

    # Highlight top-k
    palette = [
        (255, 64, 64),   # rank1 red
        (64, 255, 128),  # rank2 green
        (64, 160, 255),  # rank3 blue
    ]
    for i, c in enumerate(candidates[:max(1, top_k_highlight)]):
        pts = c.get("px", None)
        if pts is None or len(pts) < 2:
            continue
        xy = [(int(p[0]), int(p[1])) for p in pts]
        color = palette[i] if i < len(palette) else (255, 220, 64)
        width = max(3, 6 - i)
        draw.line(xy, fill=color, width=width)

    # dynamic legend with maneuvers, updated every frame
    color_names = ["red", "green", "blue"]
    legend_parts = []
    for i, c in enumerate(candidates[:max(1, top_k_highlight)]):
        m = str(c.get("m", "?"))
        cname = color_names[i] if i < len(color_names) else "yellow"
        legend_parts.append(f"#{i+1} {cname} {m}")
    legend = "Top: " + "  ".join(legend_parts) if legend_parts else "Top:"
    legend_lines = textwrap.wrap(legend, width=56)[:3]

    lx, ly = 12, 10
    line_h = 16
    max_w = 120
    if font_legend is not None:
        try:
            bb_h = font_legend.getbbox("Ag")
            line_h = max(14, int(bb_h[3] - bb_h[1]) + 4)
        except Exception:
            line_h = max(14, int(getattr(font_legend, "size", 14)) + 4)
        for ln in legend_lines:
            try:
                bb = draw.textbbox((0, 0), ln, font=font_legend)
                max_w = max(max_w, int(bb[2] - bb[0]))
            except Exception:
                pass
    box_h = line_h * max(1, len(legend_lines)) + 6
    draw.rectangle([lx - 4, ly - 3, lx + max_w + 6, ly + box_h], fill=(96, 96, 96))
    yy = ly
    for ln in legend_lines:
        draw.text((lx, yy), ln, fill=(255, 255, 255), font=font_legend)
        yy += line_h

    return np.asarray(img)


def world_to_local_topdown_px(
    pts_xy: np.ndarray,
    ego_xy: np.ndarray,
    size: int = 300,
    meters: float = 60.0
) -> Optional[np.ndarray]:
    """
    Fallback mapping: local topdown centered at ego (NOT tied to MetaDrive renderer)
    """
    if pts_xy is None or len(pts_xy) == 0:
        return None
    pts = np.asarray(pts_xy, dtype=np.float32)
    ego = np.asarray(ego_xy, dtype=np.float32)

    rel = pts - ego[None, :]
    s = size / meters
    px = rel[:, 0] * s + size / 2.0
    py = -rel[:, 1] * s + size / 2.0
    out = np.stack([px, py], axis=1)
    return out
