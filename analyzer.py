"""
analyzer.py – Extract visual "DNA" from viral thumbnails.

Instead of copying a thumbnail, we extract its repeatable system:
  - Emotion type   (shock, curiosity, fear, excitement, mystery)
  - Layout pattern (subject_left/right/center + text zone)
  - Contrast style (split, high, low)
  - Color DNA      (dominant palette, background color, accent color)
  - Visual hierarchy (where the eye travels first)

This DNA is stored in manifest.json and drives the smart generator.

Usage:
    python analyzer.py                   # analyze all un-analyzed active templates
    python analyzer.py --template <id>   # analyze one
    python analyzer.py --force           # re-analyze everything
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

import manifest as mf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Color helpers ──────────────────────────────────────────────────────────────

def _dominant_colors(img_rgb: np.ndarray, k: int = 6) -> list[list[int]]:
    """Return top-k dominant colors sorted by cluster size (RGB lists)."""
    pixels = np.float32(img_rgb.reshape(-1, 3))
    sample = pixels[:: max(1, len(pixels) // 5000)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(
        sample, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    idx = np.argsort(counts)[::-1]
    return centers[idx].clip(0, 255).astype(int).tolist()


def _color_temp(img_rgb: np.ndarray) -> str:
    r, g, b = float(img_rgb[:, :, 0].mean()), float(img_rgb[:, :, 1].mean()), float(img_rgb[:, :, 2].mean())
    if r > b + 18:
        return "warm"
    if b > r + 18:
        return "cool"
    return "neutral"


def _brightness_zone(gray: np.ndarray, zone: str) -> float:
    h, w = gray.shape
    if zone == "left":
        return float(gray[:, : w // 2].mean())
    if zone == "right":
        return float(gray[:, w // 2 :].mean())
    if zone == "top":
        return float(gray[: h // 2, :].mean())
    return float(gray[h // 2 :, :].mean())  # bottom


# ── Emotion classification ─────────────────────────────────────────────────────

def _classify_emotion(img_rgb: np.ndarray, dominant_colors: list) -> str:
    """
    Heuristic emotion from palette + contrast.

    Rules (ordered by priority):
      red/orange dominant + high contrast  → shock / danger
      yellow dominant + high brightness    → excitement / celebration
      dark palette + low brightness        → mystery / fear
      cool blues + medium contrast         → curiosity / tech
      high saturation overall              → excitement
      else                                 → curiosity
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    avg_brightness = float(gray.mean())
    contrast = float(gray.std())

    if not dominant_colors:
        return "curiosity"

    top_r, top_g, top_b = dominant_colors[0]
    is_red_orange = top_r > 160 and top_g < 120 and top_b < 100
    is_yellow = top_r > 180 and top_g > 160 and top_b < 100
    is_dark = avg_brightness < 60
    is_cool_blue = top_b > top_r + 30

    if is_red_orange and contrast > 55:
        return "shock"
    if is_yellow and avg_brightness > 130:
        return "excitement"
    if is_dark and contrast < 50:
        return "fear"
    if is_cool_blue:
        return "curiosity"
    if contrast > 70:
        return "shock"
    return "curiosity"


# ── Subject / layout detection ─────────────────────────────────────────────────

def _detect_subject(img_bgr: np.ndarray) -> dict[str, Any] | None:
    """Detect the primary face and return position metadata."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    h, w = img_bgr.shape[:2]
    return {
        "x": round(fx / w, 3),
        "y": round(fy / h, 3),
        "w": round(fw / w, 3),
        "h": round(fh / h, 3),
        "cx": round((fx + fw / 2) / w, 3),
        "cy": round((fy + fh / 2) / h, 3),
    }


def _layout_from_subject(subject: dict | None) -> dict[str, str]:
    if subject is None:
        return {"subject_side": "center", "text_side": "right", "layout": "full_frame"}
    cx = subject["cx"]
    if cx < 0.42:
        return {"subject_side": "left", "text_side": "right", "layout": "subject_left_text_right"}
    if cx > 0.58:
        return {"subject_side": "right", "text_side": "left", "layout": "subject_right_text_left"}
    return {"subject_side": "center", "text_side": "right", "layout": "subject_center"}


# ── Contrast analysis ──────────────────────────────────────────────────────────

def _contrast_style(gray: np.ndarray) -> tuple[str, float]:
    h, w = gray.shape
    left_b = float(gray[:, : w // 2].mean())
    right_b = float(gray[:, w // 2 :].mean())
    lr_diff = abs(left_b - right_b) / 128.0
    overall_std = float(gray.std())

    if lr_diff > 0.30:
        return "split", round(lr_diff, 3)
    if overall_std > 65:
        return "high", round(overall_std / 128, 3)
    if overall_std < 35:
        return "low", round(overall_std / 128, 3)
    return "medium", round(overall_std / 128, 3)


# ── Visual hierarchy ───────────────────────────────────────────────────────────

def _visual_hierarchy(subject: dict | None, layout: dict) -> list[str]:
    """
    Ordered list of focal zones: where the eye travels first.
    e.g. ["face_left", "text_right", "background"]
    """
    parts = []
    if subject:
        parts.append(f"face_{layout['subject_side']}")
    parts.append(f"text_{layout['text_side']}")
    parts.append("background")
    return parts


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_thumbnail(image_path: Path) -> dict[str, Any]:
    """
    Extract and return the visual DNA dict for a single thumbnail.
    Returns {} if the image can't be read.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.warning("Cannot read image: %s", image_path)
        return {}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    dominant = _dominant_colors(img_rgb)
    subject = _detect_subject(img_bgr)
    layout = _layout_from_subject(subject)
    c_style, c_ratio = _contrast_style(gray)
    emotion = _classify_emotion(img_rgb, dominant)
    c_temp = _color_temp(img_rgb)
    hierarchy = _visual_hierarchy(subject, layout)

    return {
        "dominant_colors": dominant[:5],   # [[R,G,B], ...]
        "bg_color":        dominant[0],
        "accent_color":    dominant[1] if len(dominant) > 1 else dominant[0],
        "subject_side":    layout["subject_side"],
        "text_side":       layout["text_side"],
        "layout":          layout["layout"],
        "subject_bbox":    subject,
        "contrast_style":  c_style,
        "contrast_ratio":  c_ratio,
        "color_temp":      c_temp,
        "emotion":         emotion,
        "visual_hierarchy": hierarchy,
        "avg_brightness":  round(float(gray.mean()), 1),
    }


def run(template_id: str | None = None, force: bool = False) -> None:
    manifest_data = mf.load()
    entries = (
        [mf.get_entry(manifest_data, template_id)]
        if template_id
        else mf.list_active(manifest_data)
    )
    entries = [e for e in entries if e]

    done = skip = fail = 0
    for entry in entries:
        if not force and entry.get("dna"):
            skip += 1
            continue
        original = Path(entry["files"]["original"])
        if not original.exists():
            logger.warning("Missing original for %s", entry["template_id"])
            fail += 1
            continue
        dna = analyze_thumbnail(original)
        if not dna:
            fail += 1
            continue
        entry["dna"] = dna
        manifest_data["templates"][entry["template_id"]] = entry
        logger.info(
            "Analyzed %s → emotion=%s layout=%s contrast=%s",
            entry["template_id"], dna["emotion"], dna["layout"], dna["contrast_style"],
        )
        done += 1

    mf.save(manifest_data)
    logger.info("Analysis done: %d analyzed, %d skipped, %d failed.", done, skip, fail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViralThumb analyzer")
    parser.add_argument("--template", help="Analyze a single template ID")
    parser.add_argument("--force", action="store_true", help="Re-analyze all templates")
    args = parser.parse_args()
    run(template_id=args.template, force=args.force)
