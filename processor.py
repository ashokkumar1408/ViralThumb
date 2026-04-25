"""
processor.py – The "Base-ify" Engine

For every new viral thumbnail added by curator.py, this module:
  1. Detects the primary person (protagonist) in the image.
  2. Generates a binary mask of that person.
  3. Inpaints the person out → background_only.png.
  4. Saves both mask.png and bg_only.png alongside the original.

Two backends are supported (set SEGMENTATION_BACKEND in .env):
  • 'mediapipe'  – fast, CPU-only, good enough for most thumbnails.
  • 'sam2'       – highly accurate, needs a GPU + SAM 2 checkpoint.

Usage:
    python processor.py                          # process all unprocessed templates
    python processor.py --template gaming_abc123 # process one specific template
    python processor.py --force                  # reprocess even already-done templates
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

import manifest as mf
from config import (
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    SEGMENTATION_BACKEND,
    TEMPLATES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── MediaPipe backend ──────────────────────────────────────────────────────────

def _mediapipe_mask(image_path: Path) -> np.ndarray | None:
    """Return a uint8 mask (255=person, 0=background) using MediaPipe Selfie."""
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("Install mediapipe: pip install mediapipe")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.error("Could not read %s", image_path)
        return None

    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = seg.process(img_rgb)

    if result.segmentation_mask is None:
        return None

    mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255
    # Minor morphological cleanup to remove small islands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# ── SAM 2 backend ──────────────────────────────────────────────────────────────

def _sam2_mask(image_path: Path) -> np.ndarray | None:
    """Return a uint8 mask using SAM 2 with automatic point prompting from faces."""
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "Install SAM 2: follow https://github.com/facebookresearch/segment-anything-2"
        )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    img = np.array(Image.open(image_path).convert("RGB"))

    # Detect face center as a positive prompt point
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        # Fallback: use image center as prompt
        h, w = img.shape[:2]
        input_point = np.array([[w // 2, h // 3]])
    else:
        # Largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        input_point = np.array([[x + fw // 2, y + fh // 2]])

    input_label = np.array([1])  # 1 = foreground

    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]  # shape (H, W), bool
    return (best_mask.astype(np.uint8)) * 255


# ── Inpainting ─────────────────────────────────────────────────────────────────

def inpaint_person(original_path: Path, mask: np.ndarray) -> np.ndarray:
    """
    Remove the masked person from the image using OpenCV inpainting.
    Returns a BGR image array with the person replaced by plausible background.
    """
    img_bgr = cv2.imread(str(original_path))
    if img_bgr is None:
        raise IOError(f"Cannot read {original_path}")

    # Dilate mask slightly so the inpainter covers the full silhouette edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # TELEA inpainting radius 5 gives good results without heavy blurring
    inpainted = cv2.inpaint(img_bgr, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted


# ── Per-template processing ────────────────────────────────────────────────────

def process_template(entry: dict[str, Any], force: bool = False) -> bool:
    """
    Generate mask + bg_only for a single manifest entry.
    Returns True on success.
    """
    files = entry.get("files", {})
    original_path = Path(files.get("original", ""))
    mask_path = Path(files.get("mask", ""))
    bg_only_path = Path(files.get("bg_only", ""))

    if not original_path.exists():
        logger.warning("Original not found: %s", original_path)
        return False

    if not force and mask_path.exists() and bg_only_path.exists():
        logger.debug("Already processed: %s", entry["template_id"])
        return True

    logger.info("Processing %s with backend '%s'…", entry["template_id"], SEGMENTATION_BACKEND)

    if SEGMENTATION_BACKEND == "sam2":
        mask = _sam2_mask(original_path)
    else:
        mask = _mediapipe_mask(original_path)

    if mask is None:
        logger.warning("Segmentation failed for %s — no person detected.", entry["template_id"])
        return False

    # Save mask
    cv2.imwrite(str(mask_path), mask)

    # Inpaint and save background-only version
    bg_bgr = inpaint_person(original_path, mask)
    cv2.imwrite(str(bg_only_path), bg_bgr)

    logger.info("  ✓ mask → %s", mask_path.name)
    logger.info("  ✓ bg   → %s", bg_only_path.name)
    return True


# ── Entry point ────────────────────────────────────────────────────────────────

def run(template_id: str | None = None, force: bool = False) -> None:
    manifest_data = mf.load()

    if template_id:
        entry = mf.get_entry(manifest_data, template_id)
        if not entry:
            logger.error("Template '%s' not found in manifest.", template_id)
            return
        entries = [entry]
    else:
        entries = mf.list_active(manifest_data)

    ok = fail = 0
    for entry in entries:
        if process_template(entry, force=force):
            ok += 1
        else:
            fail += 1

    logger.info("Processing complete: %d succeeded, %d failed.", ok, fail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViralThumb processor")
    parser.add_argument("--template", help="Process a single template ID")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even already-completed templates",
    )
    args = parser.parse_args()
    run(template_id=args.template, force=args.force)
