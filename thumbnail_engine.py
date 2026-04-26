"""
thumbnail_engine.py – Template library for ViralThumb.

Each template is a 1280×720 canvas with:
  • A proper viral-looking background (not a gradient)
  • A defined FACE ZONE  — where the user's extracted photo goes
  • A defined TEXT ZONE  — where headline + sub-text go

Templates are generated once and cached in .template_cache/.
"""

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
CACHE_DIR = Path(__file__).parent / ".template_cache"

# ── Font helper ────────────────────────────────────────────────────────────────

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
            pass
    return ImageFont.load_default()

# ── Color helpers ──────────────────────────────────────────────────────────────

def _lerp_c(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def _noise_layer(w, h, alpha=18):
    """Subtle film-grain noise to prevent flat-looking solids."""
    noise = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    layer = Image.fromarray(noise).convert("RGBA")
    layer.putalpha(alpha)
    return layer

def _vignette(w, h, strength=0.55) -> np.ndarray:
    cx, cy = w/2, h/2
    Y, X   = np.ogrid[:h, :w]
    d      = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
    return (1.0 - strength*np.clip((d-0.2)/0.8, 0, 1)).astype(np.float32)

def _add_vignette(img: Image.Image, strength=0.55) -> Image.Image:
    np_img = np.array(img).astype(np.float32)
    np_img *= _vignette(W, H, strength)[:, :, None]
    return Image.fromarray(np_img.clip(0, 255).astype(np.uint8))

def _starburst(draw, cx, cy, n, max_r, c1, c2):
    for i in range(n):
        a0 = math.pi*2*i/n
        a1 = math.pi*2*(i+.5)/n
        a2 = math.pi*2*(i+1)/n
        draw.polygon([
            (cx, cy),
            (cx+math.cos(a0)*max_r, cy+math.sin(a0)*max_r),
            (cx+math.cos(a1)*max_r, cy+math.sin(a1)*max_r),
            (cx+math.cos(a2)*max_r, cy+math.sin(a2)*max_r),
        ], fill=c1 if i%2==0 else c2)

def _glow_circle(img_np, cx, cy, r, color, strength=0.7):
    Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X-cx)**2 + (Y-cy)**2)
    mask = (np.clip(1-d/r, 0, 1)**2 * strength)[:, :, None]
    glow = np.array(color, dtype=np.float32)
    img_np[:] = (img_np * (1-mask) + glow * mask).clip(0, 255)

# ── Template definitions ───────────────────────────────────────────────────────
# face_zone / text_zone: pixel rects {x1,y1,x2,y2} for placement

TEMPLATES: dict[str, dict[str, Any]] = {

    "shock_classic": {
        "name": "Classic Shock",
        "preview_colors": ["#D94000", "#FF8C00", "#1A1A1A"],
        "face_side": "left",
        "face_zone":   {"x1":  20, "y1": 30, "x2": 620, "y2": 720},
        "text_zone":   {"x1": 650, "y1": 80, "x2":1240, "y2": 640},
        "text_align":  "center",
        "text_color":  (255, 255, 255),
        "text_stroke": (160, 20, 0),
        "outline_color": (255, 235, 0),
        "glow_color":    (255, 140, 0),
    },
    "dark_reveal": {
        "name": "Dark Reveal",
        "preview_colors": ["#040410", "#0A0830", "#005AFF"],
        "face_side": "right",
        "face_zone":   {"x1": 580, "y1": 20, "x2":1260, "y2": 720},
        "text_zone":   {"x1":  40, "y1": 80, "x2": 560, "y2": 640},
        "text_align":  "left",
        "text_color":  (255, 255, 255),
        "text_stroke": (0, 30, 120),
        "outline_color": (0, 210, 255),
        "glow_color":    (0, 100, 255),
    },
    "golden_win": {
        "name": "Golden Win",
        "preview_colors": ["#0A0A0A", "#1E1500", "#C8A000"],
        "face_side": "left",
        "face_zone":   {"x1":  10, "y1": 20, "x2": 640, "y2": 720},
        "text_zone":   {"x1": 660, "y1": 60, "x2":1250, "y2": 660},
        "text_align":  "center",
        "text_color":  (255, 220, 0),
        "text_stroke": (60, 45, 0),
        "outline_color": (230, 185, 0),
        "glow_color":    (180, 120, 0),
    },
    "red_alert": {
        "name": "Red Alert",
        "preview_colors": ["#8B0000", "#CC0020", "#FF2200"],
        "face_side": "right",
        "face_zone":   {"x1": 600, "y1":  10, "x2":1270, "y2": 720},
        "text_zone":   {"x1":  30, "y1": 60, "x2": 580, "y2": 660},
        "text_align":  "left",
        "text_color":  (255, 255, 255),
        "text_stroke": (100, 0, 0),
        "outline_color": (255, 255, 255),
        "glow_color":    (255, 80, 0),
    },
    "gaming_pro": {
        "name": "Gaming Pro",
        "preview_colors": ["#03030F", "#0A0520", "#00E5FF"],
        "face_side": "right",
        "face_zone":   {"x1": 560, "y1": 10, "x2":1270, "y2": 720},
        "text_zone":   {"x1":  30, "y1": 70, "x2": 540, "y2": 650},
        "text_align":  "left",
        "text_color":  (0, 230, 255),
        "text_stroke": (0, 20, 80),
        "outline_color": (0, 230, 255),
        "glow_color":    (0, 80, 200),
    },
    "yellow_hype": {
        "name": "Yellow Hype",
        "preview_colors": ["#FFD000", "#FFA000", "#111111"],
        "face_side": "right",
        "face_zone":   {"x1": 560, "y1": 10, "x2":1270, "y2": 720},
        "text_zone":   {"x1":  20, "y1": 60, "x2": 540, "y2": 660},
        "text_align":  "left",
        "text_color":  (10, 10, 10),
        "text_stroke": (255, 255, 255),
        "outline_color": (255, 255, 255),
        "glow_color":    (255, 200, 0),
    },
    "midnight_story": {
        "name": "Midnight Story",
        "preview_colors": ["#0A0015", "#1A0040", "#8B00FF"],
        "face_side": "left",
        "face_zone":   {"x1":  20, "y1": 20, "x2": 620, "y2": 720},
        "text_zone":   {"x1": 650, "y1": 70, "x2":1250, "y2": 650},
        "text_align":  "center",
        "text_color":  (255, 255, 255),
        "text_stroke": (80, 0, 160),
        "outline_color": (200, 100, 255),
        "glow_color":    (120, 0, 255),
    },
    "money_moves": {
        "name": "Money Moves",
        "preview_colors": ["#050F05", "#0A200A", "#00C853"],
        "face_side": "left",
        "face_zone":   {"x1":  10, "y1": 20, "x2": 640, "y2": 720},
        "text_zone":   {"x1": 660, "y1": 70, "x2":1250, "y2": 650},
        "text_align":  "center",
        "text_color":  (0, 220, 80),
        "text_stroke": (0, 40, 10),
        "outline_color": (0, 220, 80),
        "glow_color":    (0, 120, 40),
    },
}


# ── Per-template background renderers ─────────────────────────────────────────

def _render_shock_classic() -> Image.Image:
    img = Image.new("RGB", (W, H), (20, 10, 5))
    draw = ImageDraw.Draw(img)
    maxr = math.sqrt(W**2 + H**2)
    _starburst(draw, int(W*0.27), int(H*0.38), 28, maxr,
               (220, 55, 0), (255, 120, 10))
    # Dark right panel
    np_img = np.array(img).astype(np.float32)
    mask = np.zeros((H, W), np.float32)
    mask[:, int(W*0.50):] = 1.0
    mask = cv2.GaussianBlur(mask, (151, 151), 0)
    np_img = np_img * (1 - mask[:,:,None]*0.72)
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    # Exclamation mark watermark element
    draw = ImageDraw.Draw(img)
    f = _font(260)
    draw.text((710, 20), "!", font=f, fill=(255, 50, 0, 40))
    return _add_vignette(img, 0.40)


def _render_dark_reveal() -> Image.Image:
    np_img = np.zeros((H, W, 3), np.float32)
    np_img[:] = [4, 4, 16]
    # Blue-white spotlight from the right
    _glow_circle(np_img, int(W*0.72), int(H*0.40), int(H*0.55), [0, 80, 220], 0.55)
    _glow_circle(np_img, int(W*0.72), int(H*0.40), int(H*0.22), [40, 120, 255], 0.40)
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    # Horizontal scan lines
    draw = ImageDraw.Draw(img)
    for y in range(0, H, 14):
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, 12))
    # Left zone darkener
    np_img2 = np.array(img).astype(np.float32)
    mask = np.zeros((H, W), np.float32)
    mask[:, :int(W*0.46)] = 0.45
    mask = cv2.GaussianBlur(mask, (201, 201), 0)
    np_img2 *= (1-mask[:,:,None])
    img = Image.fromarray(np_img2.clip(0,255).astype(np.uint8))
    return _add_vignette(img, 0.50)


def _render_golden_win() -> Image.Image:
    img = Image.new("RGB", (W, H), (8, 8, 8))
    draw = ImageDraw.Draw(img)
    # Gold diagonal accent on the right
    draw.polygon([(int(W*0.55), 0), (W, 0), (W, H), (int(W*0.48), H)],
                 fill=(28, 20, 0))
    np_img = np.array(img).astype(np.float32)
    # Gold glow on right side
    _glow_circle(np_img, int(W*0.82), int(H*0.40), int(H*0.6), [180, 130, 0], 0.45)
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # Thin gold border lines
    for offset, alpha_val in [(3,80), (7,50), (12,25)]:
        poly = [(int(W*0.55)-offset, 0), (W, 0), (W, H), (int(W*0.48)-offset, H)]
        # draw transparent lines manually via numpy
    np_img = np.array(img).astype(np.float32)
    for y in range(H):
        t = y/H
        xb = int(W*0.55 - t*W*0.07)
        np_img[y, max(0,xb-3):xb+3] = [200, 160, 0]
    # Star elements
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    f = _font(80)
    draw.text((680, 40), "★", font=f, fill=(200, 160, 0, 60))
    draw.text((1100, 580), "★", font=f, fill=(200, 160, 0, 40))
    return _add_vignette(img, 0.45)


def _render_red_alert() -> Image.Image:
    img = Image.new("RGB", (W, H), (140, 0, 0))
    draw = ImageDraw.Draw(img)
    maxr = math.sqrt(W**2 + H**2)
    _starburst(draw, int(W*0.72), int(H*0.38), 22, maxr,
               (180, 0, 20), (220, 10, 0))
    # Dark left panel
    np_img = np.array(img).astype(np.float32)
    mask = np.zeros((H, W), np.float32)
    mask[:, :int(W*0.50)] = 0.68
    mask = cv2.GaussianBlur(mask, (151, 151), 0)
    np_img *= (1-mask[:,:,None])
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    # Warning stripes at top
    draw = ImageDraw.Draw(img)
    stripe_h = 28
    for x in range(0, W, stripe_h*2):
        draw.polygon([(x,0),(x+stripe_h,0),(x+stripe_h,stripe_h),(x,stripe_h)],
                     fill=(255,200,0))
    for x in range(0, W, stripe_h*2):
        draw.polygon([(x,H-stripe_h),(x+stripe_h,H-stripe_h),(x+stripe_h,H),(x,H)],
                     fill=(255,200,0))
    return _add_vignette(img, 0.42)


def _render_gaming_pro() -> Image.Image:
    np_img = np.zeros((H, W, 3), np.float32)
    np_img[:] = [3, 3, 15]
    # Neon right bloom
    _glow_circle(np_img, int(W*0.76), int(H*0.45), int(H*0.65), [0, 200, 255], 0.50)
    _glow_circle(np_img, int(W*0.76), int(H*0.45), int(H*0.20), [80, 240, 255], 0.35)
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # Hex grid overlay on left
    hex_r = 36
    for row in range(-1, 12):
        for col in range(-1, 8):
            cx = col * hex_r * 1.73 + (row%2)*hex_r*0.87
            cy = row * hex_r * 1.5
            pts = [(cx+hex_r*math.cos(math.pi/180*(60*i+30)),
                    cy+hex_r*math.sin(math.pi/180*(60*i+30))) for i in range(6)]
            draw.polygon(pts, outline=(0, 80, 120, 30), fill=None)
    # Horizontal accent lines
    for y_pos, alpha in [(80,35),(160,25),(H-80,35),(H-160,25)]:
        draw.line([(0,y_pos),(int(W*0.46),y_pos)], fill=(0,200,255,alpha), width=2)
    # Left darkener
    np_img2 = np.array(img).astype(np.float32)
    mask = np.zeros((H,W), np.float32)
    mask[:,:int(W*0.46)] = 0.35
    mask = cv2.GaussianBlur(mask, (201,201), 0)
    np_img2 *= (1-mask[:,:,None])
    return _add_vignette(Image.fromarray(np_img2.clip(0,255).astype(np.uint8)), 0.48)


def _render_yellow_hype() -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 200, 0))
    draw = ImageDraw.Draw(img)
    maxr = math.sqrt(W**2 + H**2)
    _starburst(draw, int(W*0.73), int(H*0.38), 26, maxr,
               (255, 170, 0), (255, 210, 10))
    # Dark left panel for text
    np_img = np.array(img).astype(np.float32)
    mask = np.zeros((H,W), np.float32)
    mask[:, :int(W*0.46)] = 0.80
    mask = cv2.GaussianBlur(mask, (151,151), 0)
    np_img *= (1-mask[:,:,None])
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    return _add_vignette(img, 0.38)


def _render_midnight_story() -> Image.Image:
    np_img = np.zeros((H,W,3), np.float32)
    np_img[:] = [10, 0, 21]
    _glow_circle(np_img, int(W*0.26), int(H*0.42), int(H*0.60), [100, 0, 220], 0.55)
    _glow_circle(np_img, int(W*0.26), int(H*0.42), int(H*0.22), [160, 60, 255], 0.35)
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    # Star particles
    draw = ImageDraw.Draw(img)
    rng = random.Random(42)
    for _ in range(120):
        sx = rng.randint(600, W-20)
        sy = rng.randint(10, H-10)
        r  = rng.randint(1, 3)
        a  = rng.randint(60, 200)
        draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(255,255,255,a))
    # Right darkener
    np_img2 = np.array(img).astype(np.float32)
    mask = np.zeros((H,W), np.float32)
    mask[:, int(W*0.52):] = 0.42
    mask = cv2.GaussianBlur(mask, (201,201), 0)
    np_img2 *= (1-mask[:,:,None])
    return _add_vignette(Image.fromarray(np_img2.clip(0,255).astype(np.uint8)), 0.46)


def _render_money_moves() -> Image.Image:
    img = Image.new("RGB", (W,H), (5, 15, 5))
    draw = ImageDraw.Draw(img)
    np_img = np.array(img).astype(np.float32)
    _glow_circle(np_img, int(W*0.26), int(H*0.42), int(H*0.58), [0, 150, 40], 0.50)
    img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # $ symbol watermark
    f = _font(400)
    draw.text((500, -80), "$", font=f, fill=(0, 80, 10, 25))
    # Right dark panel
    np_img2 = np.array(img).astype(np.float32)
    mask = np.zeros((H,W), np.float32)
    mask[:, int(W*0.52):] = 0.70
    mask = cv2.GaussianBlur(mask, (151,151), 0)
    np_img2 *= (1-mask[:,:,None])
    # Grid lines on right
    img2 = Image.fromarray(np_img2.clip(0,255).astype(np.uint8))
    d2 = ImageDraw.Draw(img2)
    for y in range(0, H, 60):
        d2.line([(int(W*0.52), y), (W, y)], fill=(0,80,20,20), width=1)
    for x in range(int(W*0.52), W, 80):
        d2.line([(x,0),(x,H)], fill=(0,80,20,20), width=1)
    return _add_vignette(img2, 0.48)


_RENDERERS = {
    "shock_classic":   _render_shock_classic,
    "dark_reveal":     _render_dark_reveal,
    "golden_win":      _render_golden_win,
    "red_alert":       _render_red_alert,
    "gaming_pro":      _render_gaming_pro,
    "yellow_hype":     _render_yellow_hype,
    "midnight_story":  _render_midnight_story,
    "money_moves":     _render_money_moves,
}


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(template_id: str) -> Path:
    return CACHE_DIR / f"{template_id}.png"


def get_background(template_id: str) -> Image.Image:
    """Return (and cache) the background for a template."""
    CACHE_DIR.mkdir(exist_ok=True)
    cp = _cache_path(template_id)
    if cp.exists():
        return Image.open(cp).convert("RGB")
    renderer = _RENDERERS.get(template_id)
    if not renderer:
        raise ValueError(f"Unknown template: {template_id}")
    bg = renderer()
    bg.save(str(cp))
    return bg


def list_templates() -> list[dict]:
    return [
        {
            "id":   tid,
            "name": TEMPLATES[tid]["name"],
            "colors": TEMPLATES[tid]["preview_colors"],
            "face_side": TEMPLATES[tid]["face_side"],
        }
        for tid in TEMPLATES
    ]


def pre_render_all() -> None:
    """Generate and cache all template backgrounds."""
    for tid in TEMPLATES:
        path = _cache_path(tid)
        if not path.exists():
            get_background(tid)
