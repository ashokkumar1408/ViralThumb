"""
generator.py – Smart Thumbnail Rebuilder

Core function:
    generate_from_template(template_id, user_image_path, output_path=None)

Algorithm (no InsightFace, no broken inpainting):
  1. Load the template's visual DNA from manifest.json.
  2. Extract user subject using rembg (clean AI background removal).
  3. Build a FRESH background from the template's color DNA —
     never copy the original creator's background.
  4. Place the user cutout in the same visual zone as the original subject.
  5. Apply color grading and vignette to match the template's contrast style.

Result: a fresh, original thumbnail that inherits the viral visual system
(layout, contrast, emotion, color temperature) without cloning anyone's content.

CLI:
    python generator.py gaming_abc123 path/to/photo.jpg
    python generator.py gaming_abc123 photo.jpg --output output/my_thumb.jpg
"""

import argparse
import io
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import manifest as mf
from config import OUTPUT_DIR, THUMBNAIL_HEIGHT as H, THUMBNAIL_WIDTH as W

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Step 1: Extract user subject with rembg ────────────────────────────────────

def _extract_cutout(image_path: Path) -> Image.Image:
    """Remove background from user photo. Returns RGBA PIL Image."""
    try:
        from rembg import remove as rembg_remove
    except ImportError:
        raise ImportError("Install rembg: pip install rembg")

    with open(image_path, "rb") as f:
        data = f.read()
    result = rembg_remove(data)
    return Image.open(io.BytesIO(result)).convert("RGBA")


# ── Step 2: Build a fresh background ──────────────────────────────────────────

def _clamp_color(c: list | tuple) -> tuple:
    return tuple(min(255, max(0, int(v))) for v in c[:3])


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(c1, c2, t: float) -> tuple:
    return tuple(int(_lerp(c1[i], c2[i], t)) for i in range(3))


def _adjust(color, factor: float) -> tuple:
    return _clamp_color([c * factor for c in color[:3]])


def _build_background(dna: dict) -> Image.Image:
    """
    Create a fresh 1280×720 canvas based on template DNA.

    Layout strategies:
      subject_left_text_right  → accent panel left, dark panel right
      subject_right_text_left  → dark panel left, accent panel right
      subject_center           → radial gradient, darker edges
      full_frame               → full gradient, no panels
    """
    bg_rgb    = _clamp_color(dna.get("bg_color",     [25, 25, 40]))
    acc_rgb   = _clamp_color(dna.get("accent_color", [60, 60, 120]))
    subj_side = dna.get("subject_side", "center")
    layout    = dna.get("layout", "subject_center")

    canvas = Image.new("RGB", (W, H))
    draw   = ImageDraw.Draw(canvas)

    if subj_side in ("left", "right"):
        # Split-panel gradient
        if subj_side == "left":
            left_color  = _adjust(acc_rgb, 1.25)
            right_color = _adjust(bg_rgb,  0.50)
        else:
            left_color  = _adjust(bg_rgb,  0.50)
            right_color = _adjust(acc_rgb, 1.25)

        for x in range(W):
            t = x / (W - 1)
            # S-curve for a punchy split rather than a slow fade
            t_s = t * t * (3 - 2 * t)
            draw.line([(x, 0), (x, H)], fill=_lerp_color(left_color, right_color, t_s))

        # Darken the text side further so copy will pop
        text_start = int(W * 0.50) if subj_side == "left" else 0
        text_end   = W              if subj_side == "left" else int(W * 0.50)
        dark_layer = np.zeros((H, W), dtype=np.float32)
        dark_layer[:, text_start:text_end] = 0.40
        dark_layer = cv2.GaussianBlur(dark_layer, (121, 121), 0)
        canvas_np  = np.array(canvas).astype(np.float32)
        canvas_np *= (1.0 - dark_layer[:, :, np.newaxis])
        canvas = Image.fromarray(canvas_np.clip(0, 255).astype(np.uint8))

    else:
        # Center layout: top-to-bottom gradient
        top    = _adjust(bg_rgb,  0.55)
        bottom = _adjust(acc_rgb, 1.15)
        for y in range(H):
            t = y / (H - 1)
            draw.line([(0, y), (W, y)], fill=_lerp_color(top, bottom, t))

    # Subtle diagonal accent stripe (adds visual depth)
    stripe_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stripe_img)
    stripe_color = (*_adjust(acc_rgb, 1.5), 18)
    for offset in range(-H, W, 80):
        sd.polygon(
            [(offset, 0), (offset + 55, 0), (offset + 55 + H, H), (offset + H, H)],
            fill=stripe_color,
        )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), stripe_img).convert("RGB")

    return canvas


# ── Step 3: Place user cutout ──────────────────────────────────────────────────

def _place_subject(canvas: Image.Image, cutout: Image.Image, dna: dict) -> Image.Image:
    """
    Scale and position the user cutout in the template's subject zone.
    Feathers the bottom edge so the subject blends into the floor.
    """
    subj_side = dna.get("subject_side", "center")

    # Scale: fill 85% of canvas height, cap at 52% of canvas width
    tgt_h = int(H * 0.87)
    aspect = cutout.width / max(cutout.height, 1)
    tgt_w  = int(tgt_h * aspect)
    if tgt_w > int(W * 0.52):
        tgt_w = int(W * 0.52)
        tgt_h = int(tgt_w / max(aspect, 0.01))

    cutout_r = cutout.resize((tgt_w, tgt_h), Image.LANCZOS)

    # Vertical: bottom-align
    y = H - tgt_h - int(H * 0.01)

    # Horizontal: keep subject in its original side
    margin = int(W * 0.02)
    if subj_side == "left":
        x = margin
    elif subj_side == "right":
        x = W - tgt_w - margin
    else:
        x = (W - tgt_w) // 2

    # Feather bottom 18% of the cutout so it doesn't have a hard floor edge
    arr = np.array(cutout_r, dtype=np.float32)
    alpha = arr[:, :, 3]
    feather = int(tgt_h * 0.18)
    ramp = np.linspace(1.0, 0.0, feather, dtype=np.float32)
    alpha[-feather:] *= ramp[:, np.newaxis]
    arr[:, :, 3] = alpha.clip(0, 255)
    cutout_r = Image.fromarray(arr.astype(np.uint8))

    result = canvas.convert("RGBA")
    result.paste(cutout_r, (x, y), cutout_r)
    return result.convert("RGB")


# ── Step 4: Apply grading and vignette ────────────────────────────────────────

def _vignette(w: int, h: int, strength: float = 0.50) -> np.ndarray:
    cx, cy = w / 2, h / 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    v      = 1.0 - strength * np.clip((dist - 0.25) / 0.75, 0, 1)
    return v.astype(np.float32)


def _apply_grading(canvas: Image.Image, dna: dict) -> Image.Image:
    """
    Apply:
      • Contrast boost scaled to the template's contrast_style
      • Saturation boost for 'warm' thumbnails
      • Edge vignette
    """
    c_style  = dna.get("contrast_style", "medium")
    c_temp   = dna.get("color_temp", "neutral")
    emotion  = dna.get("emotion", "curiosity")

    # Contrast
    c_factor = {"split": 1.40, "high": 1.35, "medium": 1.20, "low": 1.10}.get(c_style, 1.20)
    canvas = ImageEnhance.Contrast(canvas).enhance(c_factor)

    # Saturation — shock/excitement thumbnails are always punchy
    s_factor = 1.35 if emotion in ("shock", "excitement") else 1.15
    canvas = ImageEnhance.Color(canvas).enhance(s_factor)

    # Slight warmth tint for warm templates
    if c_temp == "warm":
        np_img = np.array(canvas).astype(np.float32)
        np_img[:, :, 0] = np.clip(np_img[:, :, 0] * 1.06, 0, 255)  # boost R
        np_img[:, :, 2] = np.clip(np_img[:, :, 2] * 0.94, 0, 255)  # dampen B
        canvas = Image.fromarray(np_img.astype(np.uint8))

    # Sharpness
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.25)

    # Vignette
    np_img = np.array(canvas).astype(np.float32)
    v = _vignette(W, H, strength=0.45)
    np_img *= v[:, :, np.newaxis]
    canvas = Image.fromarray(np_img.clip(0, 255).astype(np.uint8))

    return canvas


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_from_template(
    template_id: str,
    user_image_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Generate a fresh thumbnail by rebuilding the viral visual system
    from DNA — not by cloning the original creator's content.

    Args:
        template_id:     Key in manifest.json.
        user_image_path: User's photo (any format supported by PIL).
        output_path:     Where to save. Auto-generated in output/ if None.

    Returns:
        Path to the saved thumbnail.
    """
    manifest_data = mf.load()
    entry = mf.get_entry(manifest_data, template_id)
    if not entry:
        raise ValueError(f"Template '{template_id}' not found in manifest.")
    if entry["status"] != "active":
        raise ValueError(f"Template '{template_id}' is not active.")

    dna = entry.get("dna") or {}
    user_image_path = Path(user_image_path)
    if not user_image_path.exists():
        raise FileNotFoundError(f"User image not found: {user_image_path}")

    logger.info(
        "Generating for template %s (emotion=%s layout=%s contrast=%s)",
        template_id,
        dna.get("emotion", "?"),
        dna.get("layout", "?"),
        dna.get("contrast_style", "?"),
    )

    # Pipeline
    cutout = _extract_cutout(user_image_path)
    canvas = _build_background(dna)
    canvas = _place_subject(canvas, cutout, dna)
    canvas = _apply_grading(canvas, dna)

    # Save
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = OUTPUT_DIR / f"{template_id}_{ts}.jpg"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), quality=95)
    logger.info("Saved → %s", output_path)
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a thumbnail from a template's DNA.")
    parser.add_argument("template_id", help="Template ID from manifest.json")
    parser.add_argument("user_image",  help="Path to the user's photo")
    parser.add_argument("--output",    help="Output path (optional)")
    args = parser.parse_args()

    out = generate_from_template(args.template_id, args.user_image, args.output)
    print(f"Generated: {out}")
