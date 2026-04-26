"""
generator.py  –  Viral YouTube Thumbnail Engine

Just call:
    generate_thumbnail("shock", "my_photo.jpg", "I CAN'T BELIEVE", "THIS ACTUALLY WORKS")

Pipeline
────────
1. rembg removes the background from the user's photo  →  clean RGBA cutout
2. Soft outer glow  +  hard coloured outline drawn around the silhouette
3. Fresh background built from a style preset:
     "burst"       → classic starburst / sunray (the #1 viral background)
     "radial_dark" → dark with a neon glow bloom
     "dark_panel"  → dark + accent colour panel
4. Subject placed in the correct visual zone (left / right / centre)
5. Bold headline + sub-text rendered with stroke
6. Saturation / contrast / vignette grade applied

Styles: shock | hype | dark_tech | gold | red_alert | viral_blue
"""

import io
import math
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from config import OUTPUT_DIR, THUMBNAIL_HEIGHT as H, THUMBNAIL_WIDTH as W

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Style presets ──────────────────────────────────────────────────────────────

STYLES: dict[str, dict] = {
    "shock": {
        "name": "Shock",
        "preview": ["#D21E00", "#FF6E00", "#FFD700"],
        "bg_type": "burst",
        "bg_primary":        (210, 30,  0),
        "bg_secondary":      (255, 110, 0),
        "bg_tertiary":       (255, 200, 0),
        "subject_side":      "left",
        "outline_color":     (255, 235, 0),
        "glow_color":        (255, 140, 0),
        "outline_thickness": 16,
        "text_side":         "right",
        "text_color":        (255, 255, 255),
        "text_stroke":       (130, 0,   0),
        "text_panel":        (0,   0,   0),
        "panel_alpha":       0.55,
    },
    "hype": {
        "name": "Hype",
        "preview": ["#FFC800", "#FF9600", "#FF5000"],
        "bg_type": "burst",
        "bg_primary":        (255, 200, 0),
        "bg_secondary":      (255, 150, 0),
        "bg_tertiary":       (255,  80, 0),
        "subject_side":      "right",
        "outline_color":     (255, 255, 255),
        "glow_color":        (255, 180, 0),
        "outline_thickness": 18,
        "text_side":         "left",
        "text_color":        (10,  10,  10),
        "text_stroke":       (255, 255, 255),
        "text_panel":        None,
        "panel_alpha":       0,
    },
    "dark_tech": {
        "name": "Dark Tech",
        "preview": ["#040414", "#0F083A", "#0064DC"],
        "bg_type": "radial_dark",
        "bg_primary":        (4,   4,   20),
        "bg_secondary":      (15,  8,   55),
        "bg_tertiary":       (0,  100, 220),
        "subject_side":      "right",
        "outline_color":     (0,  210, 255),
        "glow_color":        (0,  100, 255),
        "outline_thickness": 14,
        "text_side":         "left",
        "text_color":        (255, 255, 255),
        "text_stroke":       (0,   40, 140),
        "text_panel":        None,
        "panel_alpha":       0,
    },
    "gold": {
        "name": "Gold",
        "preview": ["#080808", "#1C1400", "#D2A000"],
        "bg_type": "dark_panel",
        "bg_primary":        (8,   8,   8),
        "bg_secondary":      (28,  20,  0),
        "bg_tertiary":       (210, 160, 0),
        "subject_side":      "left",
        "outline_color":     (230, 185, 0),
        "glow_color":        (180, 120, 0),
        "outline_thickness": 16,
        "text_side":         "right",
        "text_color":        (255, 220, 0),
        "text_stroke":       (60,  40,  0),
        "text_panel":        (20,  15,  0),
        "panel_alpha":       0.72,
    },
    "red_alert": {
        "name": "Red Alert",
        "preview": ["#A00000", "#D20014", "#FF1E00"],
        "bg_type": "burst",
        "bg_primary":        (160, 0,   0),
        "bg_secondary":      (210, 0,   20),
        "bg_tertiary":       (255, 30,  0),
        "subject_side":      "left",
        "outline_color":     (255, 255, 255),
        "glow_color":        (255, 80,  0),
        "outline_thickness": 14,
        "text_side":         "right",
        "text_color":        (255, 255, 255),
        "text_stroke":       (80,  0,   0),
        "text_panel":        (0,   0,   0),
        "panel_alpha":       0.60,
    },
    "viral_blue": {
        "name": "Viral Blue",
        "preview": ["#0014C8", "#003CFF", "#00B4FF"],
        "bg_type": "burst",
        "bg_primary":        (0,   20, 200),
        "bg_secondary":      (0,   60, 255),
        "bg_tertiary":       (0,  180, 255),
        "subject_side":      "right",
        "outline_color":     (255, 255, 255),
        "glow_color":        (0,  200, 255),
        "outline_thickness": 16,
        "text_side":         "left",
        "text_color":        (255, 240, 0),
        "text_stroke":       (0,   0,  120),
        "text_panel":        None,
        "panel_alpha":       0,
    },
}


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


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Background builders ────────────────────────────────────────────────────────

def _burst(s: dict) -> Image.Image:
    """Starburst / sunray — the #1 viral YouTube background."""
    side = s["subject_side"]
    fx   = int(W * 0.27) if side == "left" else int(W * 0.73)
    fy   = int(H * 0.36)

    alt  = [s["bg_secondary"], s.get("bg_tertiary", s["bg_secondary"])]
    img  = Image.new("RGB", (W, H), s["bg_primary"])
    draw = ImageDraw.Draw(img)
    n    = 26
    maxr = math.sqrt(W ** 2 + H ** 2)

    for i in range(n):
        a0 = math.pi * 2 * i / n
        a1 = math.pi * 2 * (i + 0.5) / n
        a2 = math.pi * 2 * (i + 1) / n
        draw.polygon([
            (fx, fy),
            (fx + math.cos(a0) * maxr, fy + math.sin(a0) * maxr),
            (fx + math.cos(a1) * maxr, fy + math.sin(a1) * maxr),
            (fx + math.cos(a2) * maxr, fy + math.sin(a2) * maxr),
        ], fill=alt[i % 2])

    return img


def _radial_dark(s: dict) -> Image.Image:
    """Dark background with a neon bloom on the subject side."""
    side = s["subject_side"]
    cx   = W * 0.72 if side == "right" else W * 0.28
    cy   = H * 0.38

    bg   = np.array(s["bg_primary"],  dtype=np.float32)
    glow = np.array(s.get("bg_tertiary", s["bg_secondary"]), dtype=np.float32)

    Y, X = np.ogrid[:H, :W]
    d    = np.sqrt(((X - cx) / (W * 0.55)) ** 2 + ((Y - cy) / (H * 0.55)) ** 2)
    t    = (np.clip(1.0 - d, 0, 1) ** 1.8)[:, :, np.newaxis]
    out  = (bg * (1 - t) + glow * t).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _dark_panel(s: dict) -> Image.Image:
    """Dark canvas with an accent panel on the subject side."""
    img = np.full((H, W, 3), s["bg_primary"], dtype=np.float32)
    acc = np.array(s.get("bg_tertiary", s["bg_secondary"]), dtype=np.float32)

    side = s["subject_side"]
    x0   = 0                if side == "left"  else int(W * 0.48)
    x1   = int(W * 0.52)   if side == "left"  else W
    img[:, x0:x1] = img[:, x0:x1] * 0.25 + acc * 0.18
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def _build_bg(s: dict) -> Image.Image:
    return {"burst": _burst, "radial_dark": _radial_dark}.get(s["bg_type"], _dark_panel)(s)


# ── Subject effects ────────────────────────────────────────────────────────────

def _cutout(path: Path) -> Image.Image:
    try:
        from rembg import remove as _rem
    except ImportError:
        raise ImportError("pip install rembg")
    return Image.open(io.BytesIO(_rem(path.read_bytes()))).convert("RGBA")


def _outline_glow(
    img: Image.Image,
    outline_color: tuple,
    glow_color:    tuple,
    outline_px:    int = 14,
    glow_px:       int = 32,
) -> Image.Image:
    """
    Layers (back to front):
        1. Soft blurred glow  (adds the luminous aura)
        2. Hard coloured outline  (defines the silhouette)
        3. Original subject
    This is the signature look of every viral thumbnail.
    """
    arr   = np.array(img)
    alpha = arr[:, :, 3].astype(np.uint8)
    h, w  = alpha.shape

    def _layer(color: tuple, mask: np.ndarray) -> Image.Image:
        l = np.zeros((h, w, 4), np.uint8)
        l[:, :, :3] = color[:3]
        l[:, :,  3] = mask.clip(0, 255).astype(np.uint8)
        return Image.fromarray(l)

    # Glow
    k_g      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glow_px * 2 + 1,) * 2)
    exp_g    = cv2.dilate(alpha, k_g).astype(np.float32)
    glow_raw = np.clip(exp_g - alpha.astype(np.float32), 0, 255)
    glow_m   = cv2.GaussianBlur(glow_raw, (glow_px * 2 + 1,) * 2, 0) * 0.85

    # Hard outline
    k_o   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_px * 2 + 1,) * 2)
    exp_o = cv2.dilate(alpha, k_o)
    out_m = np.clip(exp_o.astype(np.int32) - alpha.astype(np.int32), 0, 255)

    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for layer in (_layer(glow_color, glow_m), _layer(outline_color, out_m), img):
        result.paste(layer, (0, 0), layer)
    return result


def _place(canvas: Image.Image, subject: Image.Image, side: str) -> Image.Image:
    """Scale subject to 90 % canvas height, position on the correct side."""
    tgt_h  = int(H * 0.91)
    aspect = subject.width / max(subject.height, 1)
    tgt_w  = min(int(tgt_h * aspect), int(W * 0.55))
    tgt_h  = int(tgt_w / max(aspect, 0.01))

    sub_r  = subject.resize((tgt_w, tgt_h), Image.LANCZOS)

    # Feather bottom 14 % so it blends into the floor
    arr    = np.array(sub_r, np.float32)
    fh     = int(tgt_h * 0.14)
    arr[-fh:, :, 3] *= np.linspace(1.0, 0.0, fh, dtype=np.float32)[:, None]
    sub_r  = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    y = H - tgt_h
    x = (int(W * 0.01)              if side == "left"
         else W - tgt_w - int(W * 0.01) if side == "right"
         else (W - tgt_w) // 2)

    out = canvas.convert("RGBA")
    out.paste(sub_r, (x, y), sub_r)
    return out.convert("RGB")


# ── Text ───────────────────────────────────────────────────────────────────────

def _wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textbbox((0, 0), trial.upper(), font=f)[2] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text]


def _add_text(canvas: Image.Image, headline: str, sub_text: str, s: dict) -> Image.Image:
    if not headline and not sub_text:
        return canvas

    draw      = ImageDraw.Draw(canvas)
    side      = s["text_side"]
    pad       = 36
    x0        = W // 2 + pad if side == "right" else pad
    x1        = W - pad      if side == "right" else W // 2 - pad
    zone_w    = x1 - x0
    zone_cx   = (x0 + x1) // 2

    # Optional dark panel behind text
    if s.get("text_panel") and s.get("panel_alpha", 0) > 0:
        ov  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od  = ImageDraw.Draw(ov)
        pa  = int(s["panel_alpha"] * 255)
        od.rectangle([x0 - pad, 0, x1 + pad, H], fill=(*s["text_panel"], pa))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")
        draw   = ImageDraw.Draw(canvas)

    blocks: list[tuple] = []

    if headline:
        fsz = 110
        while fsz > 32:
            f    = _font(fsz)
            lns  = _wrap(draw, headline, f, zone_w)
            maxw = max(draw.textbbox((0, 0), ln.upper(), font=f)[2] for ln in lns)
            if maxw <= zone_w:
                break
            fsz -= 5
        blocks.append((_font(fsz), _wrap(draw, headline, _font(fsz), zone_w),
                        s["text_color"], s["text_stroke"], fsz))

    if sub_text:
        ssz  = max(40, (blocks[0][4] if blocks else 80) - 34)
        sf   = _font(ssz)
        blocks.append((sf, _wrap(draw, sub_text, sf, zone_w),
                        s.get("sub_color", s["text_color"]), s["text_stroke"], ssz))

    # Measure total height
    total_h = sum(
        (draw.textbbox((0, 0), "A", font=f)[3] + int(sz * 0.20)) * len(lns) + 14
        for f, lns, _, _, sz in blocks
    )

    y = max(40, (H - total_h) // 2)

    for f, lines, tc, sc, sz in blocks:
        lh = draw.textbbox((0, 0), "A", font=f)[3] + int(sz * 0.20)
        sw = max(4, int(sz * 0.07))
        for line in lines:
            bx = draw.textbbox((0, 0), line.upper(), font=f)
            x  = zone_cx - (bx[2] - bx[0]) // 2
            draw.text((x, y), line.upper(), font=f,
                      fill=tc, stroke_width=sw, stroke_fill=sc)
            y += lh
        y += 14

    return canvas


# ── Final grade ────────────────────────────────────────────────────────────────

def _grade(canvas: Image.Image) -> Image.Image:
    canvas = ImageEnhance.Contrast(canvas).enhance(1.28)
    canvas = ImageEnhance.Color(canvas).enhance(1.45)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.20)

    cx, cy = W / 2, H / 2
    Y, X   = np.ogrid[:H, :W]
    d      = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vig    = (1.0 - 0.44 * np.clip((d - 0.22) / 0.78, 0, 1)).astype(np.float32)

    out    = np.array(canvas).astype(np.float32) * vig[:, :, None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ── Public API ─────────────────────────────────────────────────────────────────

def get_styles() -> list[dict]:
    return [{"id": k, "name": v["name"], "preview": v["preview"]} for k, v in STYLES.items()]


def generate_thumbnail(
    style_id:       str,
    user_image_path: str | Path,
    headline:       str = "",
    sub_text:       str = "",
    output_path:    str | Path | None = None,
) -> Path:
    """
    Generate a viral YouTube thumbnail.

    Args:
        style_id:         One of shock | hype | dark_tech | gold | red_alert | viral_blue
        user_image_path:  User's photo (any format PIL supports).
        headline:         Main text, e.g. "I CAN'T BELIEVE"
        sub_text:         Second line, e.g. "THIS WORKS"
        output_path:      Where to save (auto-generated if None).
    """
    s = STYLES.get(style_id)
    if not s:
        raise ValueError(f"Unknown style '{style_id}'. Choose from: {list(STYLES)}")

    path = Path(user_image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    logger.info("Generating | style=%-12s headline=%s", style_id, headline or "(none)")

    subject = _outline_glow(
        _cutout(path),
        outline_color=s["outline_color"],
        glow_color=s["glow_color"],
        outline_px=s.get("outline_thickness", 14),
    )
    canvas  = _build_bg(s)
    canvas  = _place(canvas, subject, s["subject_side"])
    canvas  = _add_text(canvas, headline, sub_text, s)
    canvas  = _grade(canvas)

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = OUTPUT_DIR / f"{style_id}_{ts}.jpg"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out), quality=95)
    logger.info("Saved → %s", out)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate a viral YouTube thumbnail")
    ap.add_argument("style",       help=f"Style: {list(STYLES)}")
    ap.add_argument("user_image",  help="Path to your photo")
    ap.add_argument("--headline",  default="", help='Main text, e.g. "I CAN\'T BELIEVE"')
    ap.add_argument("--sub",       default="", help="Second line of text")
    ap.add_argument("--output",    default=None)
    a = ap.parse_args()
    print(generate_thumbnail(a.style, a.user_image, a.headline, a.sub, a.output))
