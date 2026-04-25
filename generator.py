"""
generator.py – User Generation API

Core function:
    generate_from_template(template_id, user_image_path, output_path=None)

Compositing pipeline:
  1. Load the pre-computed background_only.png from the template.
  2. Load the user's photo and detect/align their face using InsightFace.
  3. Swap/blend the user's face into the mask region.
  4. Composite the result onto the background at the correct position and scale.
  5. Save to output/.

Two compositing modes (set via mode parameter or COMPOSITING_MODE env var):
  • 'insightface'  – full face-swap using InsightFace inswapper model.
  • 'alpha_blend'  – simple alpha-composite; faster but less realistic.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

import manifest as mf
from config import (
    INSIGHTFACE_MODEL_PACK,
    OUTPUT_DIR,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CompositingMode = Literal["insightface", "alpha_blend"]
DEFAULT_MODE: CompositingMode = os.environ.get("COMPOSITING_MODE", "alpha_blend")  # type: ignore[assignment]


# ── InsightFace face-swap ──────────────────────────────────────────────────────

def _insightface_swap(
    bg_bgr: np.ndarray,
    mask: np.ndarray,
    user_img_bgr: np.ndarray,
) -> np.ndarray:
    """
    Use InsightFace inswapper to paste the user's face onto the background.
    The inswapper finds the source face from user_img_bgr and the target face
    from the inpainted background area bounded by the mask.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("Install insightface: pip install insightface onnxruntime")

    app = FaceAnalysis(name=INSIGHTFACE_MODEL_PACK, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=True, download_zip=True)

    src_faces = app.get(user_img_bgr)
    if not src_faces:
        raise ValueError("No face detected in user image.")
    src_face = sorted(src_faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)[0]

    tgt_faces = app.get(bg_bgr)
    if not tgt_faces:
        logger.warning("No face found in template background — falling back to alpha blend.")
        return None  # type: ignore[return-value]

    # Swap the largest face in the target with the user's face
    tgt_face = sorted(tgt_faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)[0]
    result = swapper.get(bg_bgr.copy(), tgt_face, src_face, paste_back=True)
    return result


# ── Alpha-blend compositing ────────────────────────────────────────────────────

def _detect_face_bbox(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest face, or None."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))  # type: ignore[return-value]


def _alpha_blend_composite(
    bg_bgr: np.ndarray,
    mask: np.ndarray,
    user_img_bgr: np.ndarray,
) -> np.ndarray:
    """
    Fit the user's cropped face+torso into the mask bounding box and alpha-blend.
    """
    h_bg, w_bg = bg_bgr.shape[:2]

    # Find bounding box of the mask region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("Empty mask — returning background unchanged.")
        return bg_bgr.copy()
    rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Detect & crop user around their face with generous padding
    bbox = _detect_face_bbox(user_img_bgr)
    if bbox:
        fx, fy, fw, fh = bbox
        pad_x = int(fw * 1.5)
        pad_y = int(fh * 2.5)
        x1 = max(fx - pad_x, 0)
        y1 = max(fy - pad_y, 0)
        x2 = min(fx + fw + pad_x, user_img_bgr.shape[1])
        y2 = min(fy + fh + pad_y, user_img_bgr.shape[0])
        user_crop = user_img_bgr[y1:y2, x1:x2]
    else:
        user_crop = user_img_bgr

    # Resize to fill the mask bounding box
    user_resized = cv2.resize(user_crop, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

    # Soft alpha edge from the mask ROI
    mask_roi = mask[ry : ry + rh, rx : rx + rw].astype(np.float32) / 255.0
    mask_roi = cv2.GaussianBlur(mask_roi, (21, 21), 0)
    alpha = mask_roi[:, :, np.newaxis]

    bg_copy = bg_bgr.copy()
    bg_roi = bg_copy[ry : ry + rh, rx : rx + rw].astype(np.float32)
    user_float = user_resized.astype(np.float32)

    blended = (user_float * alpha + bg_roi * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    bg_copy[ry : ry + rh, rx : rx + rw] = blended
    return bg_copy


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_from_template(
    template_id: str,
    user_image_path: str | Path,
    output_path: str | Path | None = None,
    mode: CompositingMode = DEFAULT_MODE,
) -> Path:
    """
    Composite user_image onto the named template.

    Args:
        template_id:     Key from manifest.json (e.g. "gaming_abc123").
        user_image_path: Path to the user's photo (jpg/png).
        output_path:     Where to save the result. Auto-generated if None.
        mode:            'insightface' or 'alpha_blend'.

    Returns:
        Path to the generated thumbnail.
    """
    manifest_data = mf.load()
    entry = mf.get_entry(manifest_data, template_id)
    if not entry:
        raise ValueError(f"Template '{template_id}' not found in manifest.")
    if entry["status"] != "active":
        raise ValueError(f"Template '{template_id}' is not active (status={entry['status']}).")

    files = entry["files"]
    bg_path = Path(files["bg_only"])
    mask_path = Path(files["mask"])
    user_image_path = Path(user_image_path)

    for p, label in [(bg_path, "bg_only"), (mask_path, "mask"), (user_image_path, "user_image")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    bg_bgr = cv2.imread(str(bg_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    user_bgr = cv2.imread(str(user_image_path))

    if any(x is None for x in (bg_bgr, mask, user_bgr)):
        raise IOError("Could not read one or more images.")

    # Ensure all assets share the same canvas size
    bg_bgr = cv2.resize(bg_bgr, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
    mask = cv2.resize(mask, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), interpolation=cv2.INTER_NEAREST)

    logger.info("Generating thumbnail (mode=%s, template=%s)…", mode, template_id)

    if mode == "insightface":
        result = _insightface_swap(bg_bgr, mask, user_bgr)
        if result is None:
            logger.warning("InsightFace swap failed — falling back to alpha_blend.")
            result = _alpha_blend_composite(bg_bgr, mask, user_bgr)
    else:
        result = _alpha_blend_composite(bg_bgr, mask, user_bgr)

    # Write output
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = OUTPUT_DIR / f"{template_id}_{ts}.jpg"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info("Saved → %s", output_path)
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a thumbnail from a template.")
    parser.add_argument("template_id", help="Template ID from manifest.json")
    parser.add_argument("user_image", help="Path to the user's photo")
    parser.add_argument("--output", help="Output path (optional)")
    parser.add_argument(
        "--mode",
        choices=["insightface", "alpha_blend"],
        default=DEFAULT_MODE,
        help="Compositing mode (default: %(default)s)",
    )
    args = parser.parse_args()

    out = generate_from_template(args.template_id, args.user_image, args.output, mode=args.mode)
    print(f"Generated: {out}")
