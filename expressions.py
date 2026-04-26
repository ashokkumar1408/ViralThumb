"""
expressions.py – Generate 4 expression variants from a single user photo.

Uses OpenCV face + eye detection to locate key facial regions, then applies
targeted geometric warping and colour treatment to simulate:

  original  – clean extraction, natural colours
  shock     – eyes wider, brows raised, jaw dropped, cool tint
  excited   – eyes bright, warm tones, high energy
  serious   – slight squint, cool desaturated, intense

The warp technique:
  1. Detect face rect + eye rects
  2. Build a pixel displacement map (cv2.remap) around each feature zone
  3. Apply colour grade on top

Returns: dict of { variant_name → PIL RGBA Image }
"""

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ── Face / feature detection ──────────────────────────────────────────────────

_FACE_CASCADE = None
_EYE_CASCADE  = None
_SMILE_CASCADE = None

def _cascades():
    global _FACE_CASCADE, _EYE_CASCADE, _SMILE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _EYE_CASCADE   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        _SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return _FACE_CASCADE, _EYE_CASCADE, _SMILE_CASCADE


def _detect_features(gray: np.ndarray) -> dict | None:
    fc, ec, sc = _cascades()
    faces = fc.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
    face_roi_gray   = gray[fy:fy+fh, fx:fx+fw]

    eyes   = ec.detectMultiScale(face_roi_gray, 1.1, 4, minSize=(20, 20))
    # approximate brow, mouth from face proportions if no explicit detection
    brow_y  = fy + int(fh * 0.12)
    eye_y   = fy + int(fh * 0.28)
    mouth_y = fy + int(fh * 0.65)
    jaw_y   = fy + int(fh * 0.88)

    # Refine eye_y from actual eye detections
    if len(eyes) >= 1:
        ey_vals = sorted(eyes, key=lambda e: e[1])  # sort by y
        eye_y   = fy + ey_vals[0][1] + ey_vals[0][3]//2

    return {
        "fx": fx, "fy": fy, "fw": fw, "fh": fh,
        "brow_y": brow_y,
        "eye_y":  eye_y,
        "mouth_y": mouth_y,
        "jaw_y":  jaw_y,
    }


# ── Displacement map helpers ──────────────────────────────────────────────────

def _identity_maps(h, w):
    mx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    my = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    return mx, my


def _gaussian_pull(my, mx, cy, cx, ry, rx, dy, dx, face_rect=None):
    """Add a Gaussian-shaped displacement centred at (cy,cx)."""
    Y, X = np.mgrid[:my.shape[0], :mx.shape[1]]
    gauss = np.exp(-0.5*((Y-cy)/ry)**2 - 0.5*((X-cx)/rx)**2)
    if face_rect:
        fx, fy, fw, fh = face_rect
        mask = np.zeros_like(gauss)
        mask[fy:fy+fh, fx:fx+fw] = 1.0
        gauss *= mask
    my -= gauss * dy
    mx -= gauss * dx


def _apply_maps(img_bgra: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    warped = np.zeros_like(img_bgra)
    for c in range(img_bgra.shape[2]):
        warped[:,:,c] = cv2.remap(img_bgra[:,:,c], mx, my,
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


# ── Expression warp functions ─────────────────────────────────────────────────

def _warp_shock(bgra: np.ndarray, feat: dict) -> np.ndarray:
    h, w = bgra.shape[:2]
    mx, my = _identity_maps(h, w)
    fx, fy, fw, fh = feat["fx"], feat["fy"], feat["fw"], feat["fh"]
    fr = (fx, fy, fw, fh)

    # Raise brows
    _gaussian_pull(my, mx, feat["brow_y"], fx+fw//2, fh*0.08, fw*0.45,  fh*0.045, 0, fr)
    # Widen upper eyes
    _gaussian_pull(my, mx, feat["eye_y"]-fh*0.04, fx+fw//2, fh*0.07, fw*0.40, fh*0.04, 0, fr)
    # Pull lower eye lid down
    _gaussian_pull(my, mx, feat["eye_y"]+fh*0.04, fx+fw//2, fh*0.07, fw*0.40, -fh*0.04, 0, fr)
    # Drop jaw
    _gaussian_pull(my, mx, feat["jaw_y"], fx+fw//2, fh*0.12, fw*0.38, -fh*0.06, 0, fr)
    # Open mouth (pull top up, bottom down)
    _gaussian_pull(my, mx, feat["mouth_y"]-fh*0.06, fx+fw//2, fh*0.08, fw*0.28, fh*0.03, 0, fr)
    _gaussian_pull(my, mx, feat["mouth_y"]+fh*0.06, fx+fw//2, fh*0.08, fw*0.28, -fh*0.04, 0, fr)

    return _apply_maps(bgra, mx, my)


def _warp_excited(bgra: np.ndarray, feat: dict) -> np.ndarray:
    h, w = bgra.shape[:2]
    mx, my = _identity_maps(h, w)
    fx, fy, fw, fh = feat["fx"], feat["fy"], feat["fw"], feat["fh"]
    fr = (fx, fy, fw, fh)

    # Raise brows slightly
    _gaussian_pull(my, mx, feat["brow_y"], fx+fw//2, fh*0.07, fw*0.42, fh*0.025, 0, fr)
    # Widen eyes (moderate)
    _gaussian_pull(my, mx, feat["eye_y"]-fh*0.03, fx+fw//2, fh*0.06, fw*0.38, fh*0.025, 0, fr)
    _gaussian_pull(my, mx, feat["eye_y"]+fh*0.03, fx+fw//2, fh*0.06, fw*0.38, -fh*0.025, 0, fr)
    # Raise mouth corners (smile)
    left_corner_x  = fx + int(fw * 0.22)
    right_corner_x = fx + int(fw * 0.78)
    _gaussian_pull(my, mx, feat["mouth_y"], left_corner_x,  fh*0.07, fw*0.12, fh*0.03, 0, fr)
    _gaussian_pull(my, mx, feat["mouth_y"], right_corner_x, fh*0.07, fw*0.12, fh*0.03, 0, fr)

    return _apply_maps(bgra, mx, my)


def _warp_serious(bgra: np.ndarray, feat: dict) -> np.ndarray:
    h, w = bgra.shape[:2]
    mx, my = _identity_maps(h, w)
    fx, fy, fw, fh = feat["fx"], feat["fy"], feat["fw"], feat["fh"]
    fr = (fx, fy, fw, fh)

    # Lower brows / furrow
    _gaussian_pull(my, mx, feat["brow_y"], fx+fw//2, fh*0.07, fw*0.44, -fh*0.020, 0, fr)
    # Slightly compress eye openings (squint)
    _gaussian_pull(my, mx, feat["eye_y"]-fh*0.03, fx+fw//2, fh*0.06, fw*0.40, -fh*0.018, 0, fr)
    _gaussian_pull(my, mx, feat["eye_y"]+fh*0.03, fx+fw//2, fh*0.06, fw*0.40,  fh*0.018, 0, fr)
    # Slight mouth press (flatten lips)
    _gaussian_pull(my, mx, feat["mouth_y"], fx+fw//2, fh*0.06, fw*0.32, -fh*0.01, 0, fr)

    return _apply_maps(bgra, mx, my)


# ── Colour grades per expression ──────────────────────────────────────────────

def _grade(pil_img: Image.Image, expression: str) -> Image.Image:
    if expression == "shock":
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.35)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.05)
        np_img  = np.array(pil_img).astype(np.float32)
        # Cool tint
        np_img[:,:,2] = np.clip(np_img[:,:,2]*1.08, 0, 255)
        np_img[:,:,0] = np.clip(np_img[:,:,0]*0.94, 0, 255)
        pil_img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    elif expression == "excited":
        pil_img = ImageEnhance.Color(pil_img).enhance(1.50)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.10)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.20)
        np_img  = np.array(pil_img).astype(np.float32)
        # Warm tint
        np_img[:,:,0] = np.clip(np_img[:,:,0]*1.06, 0, 255)
        np_img[:,:,2] = np.clip(np_img[:,:,2]*0.94, 0, 255)
        pil_img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    elif expression == "serious":
        pil_img = ImageEnhance.Color(pil_img).enhance(0.75)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.40)
        np_img  = np.array(pil_img).astype(np.float32)
        # Cold blue tint
        np_img[:,:,2] = np.clip(np_img[:,:,2]*1.12, 0, 255)
        np_img[:,:,0] = np.clip(np_img[:,:,0]*0.90, 0, 255)
        pil_img = Image.fromarray(np_img.clip(0,255).astype(np.uint8))
    else:  # original
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.15)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.30)
    return pil_img


# ── rembg extraction ──────────────────────────────────────────────────────────

def _extract(path: Path) -> Image.Image:
    try:
        from rembg import remove as _rem
    except ImportError:
        raise ImportError("pip install rembg")
    return Image.open(io.BytesIO(_rem(path.read_bytes()))).convert("RGBA")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_variants(user_image_path: Path) -> dict[str, Image.Image]:
    """
    Return 4 expression variants as RGBA PIL Images.

    Keys: original, shock, excited, serious
    """
    base_cutout = _extract(user_image_path)
    base_rgb    = base_cutout.convert("RGB")
    bgra        = np.array(base_cutout)             # H×W×4 uint8
    gray        = cv2.cvtColor(np.array(base_rgb), cv2.COLOR_RGB2GRAY)
    feat        = _detect_features(gray)

    results: dict[str, Image.Image] = {}

    warps = {
        "original": None,
        "shock":    _warp_shock,
        "excited":  _warp_excited,
        "serious":  _warp_serious,
    }

    for name, warp_fn in warps.items():
        if warp_fn is None or feat is None:
            warped_pil = base_cutout.copy()
        else:
            warped_np  = warp_fn(bgra.copy(), feat)
            warped_pil = Image.fromarray(warped_np)

        results[name] = _grade(warped_pil, name)

    return results
