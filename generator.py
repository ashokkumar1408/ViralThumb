"""
generator.py – Viral Thumbnail Compositor

combine_thumbnail(template_id, expression_variant, headline, sub_text, output_path)

Pipeline
────────
1. Load template background from thumbnail_engine
2. Load expression variant (RGBA cutout with expression applied)
3. Add coloured outline + glow around subject silhouette
4. Resize subject to fill the template's face_zone
5. Composite subject onto background
6. Render headline + sub-text in text_zone with thick stroke
7. Apply final saturation / contrast / vignette grade
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from config import OUTPUT_DIR, THUMBNAIL_HEIGHT as H, THUMBNAIL_WIDTH as W
from thumbnail_engine import TEMPLATES, get_background

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Font ───────────────────────────────────────────────────────────────────────

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Impact.ttf",
    "/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/impact.ttf",
]

def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ── Subject outline + glow ────────────────────────────────────────────────────

def _outline_glow(
    img:          Image.Image,
    outline_col:  tuple,
    glow_col:     tuple,
    outline_px:   int = 14,
    glow_px:      int = 30,
) -> Image.Image:
    arr   = np.array(img)
    alpha = arr[:, :, 3].astype(np.uint8)
    h, w  = alpha.shape

    def _layer(color, mask):
        l = np.zeros((h, w, 4), np.uint8)
        l[:,:,:3] = color[:3]
        l[:,:,3]  = mask.clip(0, 255).astype(np.uint8)
        return Image.fromarray(l)

    kg    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glow_px*2+1,)*2)
    eg    = cv2.dilate(alpha, kg).astype(np.float32)
    gm    = cv2.GaussianBlur(np.clip(eg - alpha.astype(np.float32), 0, 255),
                              (glow_px*2+1,)*2, 0) * 0.80

    ko    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_px*2+1,)*2)
    eo    = cv2.dilate(alpha, ko)
    om    = np.clip(eo.astype(np.int32) - alpha.astype(np.int32), 0, 255)

    result = Image.new("RGBA", img.size, (0,0,0,0))
    for layer in (_layer(glow_col, gm), _layer(outline_col, om), img):
        result.paste(layer, (0,0), layer)
    return result


# ── Subject placement ─────────────────────────────────────────────────────────

def _place_subject(canvas: Image.Image, subject: Image.Image, face_zone: dict) -> Image.Image:
    """
    Scale and position subject to fill face_zone.
    Feathers the bottom edge.
    """
    zw = face_zone["x2"] - face_zone["x1"]
    zh = face_zone["y2"] - face_zone["y1"]

    # Scale: fit inside zone while preserving aspect ratio
    aspect = subject.width / max(subject.height, 1)
    if aspect > zw/zh:
        new_w = zw
        new_h = int(zw / aspect)
    else:
        new_h = zh
        new_w = int(zh * aspect)

    sub_r = subject.resize((new_w, new_h), Image.LANCZOS)

    # Feather bottom 18 %
    arr  = np.array(sub_r, np.float32)
    fh   = int(new_h * 0.18)
    ramp = np.linspace(1.0, 0.0, fh, np.float32)
    arr[-fh:, :, 3] *= ramp[:, None]
    sub_r = Image.fromarray(arr.clip(0,255).astype(np.uint8))

    # Bottom-align inside zone
    px = face_zone["x1"] + (zw - new_w) // 2
    py = face_zone["y2"] - new_h

    out = canvas.convert("RGBA")
    out.paste(sub_r, (px, py), sub_r)
    return out.convert("RGB")


# ── Text rendering ────────────────────────────────────────────────────────────

def _wrap(draw, text, f, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textbbox((0,0), trial.upper(), font=f)[2] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text]


def _render_text(canvas: Image.Image, headline: str, sub_text: str, tmpl: dict) -> Image.Image:
    if not headline and not sub_text:
        return canvas

    tz    = tmpl["text_zone"]
    tz_x1, tz_y1, tz_x2, tz_y2 = tz["x1"], tz["y1"], tz["x2"], tz["y2"]
    tz_w  = tz_x2 - tz_x1
    tz_h  = tz_y2 - tz_y1
    tz_cx = (tz_x1 + tz_x2) // 2

    align = tmpl.get("text_align", "center")
    tc    = tmpl["text_color"]
    sc    = tmpl["text_stroke"]

    draw  = ImageDraw.Draw(canvas)

    blocks: list[tuple] = []

    if headline:
        fsz = 120
        while fsz > 32:
            f   = _font(fsz)
            lns = _wrap(draw, headline, f, tz_w - 20)
            mw  = max(draw.textbbox((0,0), ln.upper(), font=f)[2] for ln in lns)
            if mw <= tz_w - 20:
                break
            fsz -= 5
        blocks.append((_font(fsz), _wrap(draw, headline, _font(fsz), tz_w-20),
                        tc, sc, fsz))

    if sub_text:
        ssz = max(38, (blocks[0][4] if blocks else 80) - 36)
        sf  = _font(ssz)
        blocks.append((sf, _wrap(draw, sub_text, sf, tz_w-20),
                        tc, sc, ssz))

    # Measure total block height
    total_h = 0
    for (f, lns, _, _, sz) in blocks:
        lh = draw.textbbox((0,0), "A", font=f)[3] + int(sz*0.22)
        total_h += lh * len(lns) + 10

    y = tz_y1 + (tz_h - total_h) // 2

    for (f, lines, t_col, s_col, sz) in blocks:
        lh = draw.textbbox((0,0), "A", font=f)[3] + int(sz*0.22)
        sw = max(4, int(sz * 0.07))
        for line in lines:
            bx = draw.textbbox((0,0), line.upper(), font=f)
            lw = bx[2] - bx[0]
            if align == "left":
                x = tz_x1 + 10
            elif align == "right":
                x = tz_x2 - lw - 10
            else:
                x = tz_cx - lw // 2
            draw.text((x, y), line.upper(), font=f,
                      fill=t_col, stroke_width=sw, stroke_fill=s_col)
            y += lh
        y += 10

    return canvas


# ── Final grade ───────────────────────────────────────────────────────────────

def _grade(canvas: Image.Image) -> Image.Image:
    canvas = ImageEnhance.Contrast(canvas).enhance(1.22)
    canvas = ImageEnhance.Color(canvas).enhance(1.35)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.20)
    cx, cy = W/2, H/2
    Y, X   = np.ogrid[:H, :W]
    d      = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
    vig    = (1.0 - 0.38*np.clip((d-0.25)/0.75, 0, 1)).astype(np.float32)
    np_img = np.array(canvas).astype(np.float32) * vig[:,:,None]
    return Image.fromarray(np_img.clip(0,255).astype(np.uint8))


# ── Public API ────────────────────────────────────────────────────────────────

def combine_thumbnail(
    template_id:       str,
    subject:           Image.Image,   # RGBA cutout with expression already applied
    headline:          str = "",
    sub_text:          str = "",
    output_path:       str | Path | None = None,
) -> Path:
    """
    Compose final thumbnail: background + subject + text.

    Args:
        template_id:  Key from TEMPLATES dict in thumbnail_engine.
        subject:      RGBA PIL Image (expression variant already applied).
        headline:     Main text line.
        sub_text:     Secondary text line.
        output_path:  Save location (auto-generated if None).
    """
    tmpl = TEMPLATES.get(template_id)
    if not tmpl:
        raise ValueError(f"Unknown template: {template_id}")

    logger.info("Compositing template=%s headline=%s", template_id, headline or "(none)")

    # Outline + glow around subject
    outline_col = tmpl.get("outline_color", (255, 235, 0))
    glow_col    = tmpl.get("glow_color", (255, 180, 0))
    subject_fx  = _outline_glow(subject, outline_col, glow_col)

    bg      = get_background(template_id)
    canvas  = _place_subject(bg, subject_fx, tmpl["face_zone"])
    canvas  = _render_text(canvas, headline, sub_text, tmpl)
    canvas  = _grade(canvas)

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = OUTPUT_DIR / f"{template_id}_{ts}.jpg"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out), quality=95)
    logger.info("Saved → %s", out)
    return out
