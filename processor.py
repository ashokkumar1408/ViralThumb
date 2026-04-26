"""
processor.py – Template Preparation

Prepares downloaded thumbnails for generation by:
  1. Running rembg to verify a person is present (flags templates with no subject).
  2. Running the analyzer to extract visual DNA and store it in manifest.json.

No inpainting needed — the smart generator builds a fresh background from DNA,
so we never need the bg_only.png file at all.

Usage:
    python processor.py                          # process all unprocessed templates
    python processor.py --template gaming_abc123 # process one specific template
    python processor.py --force                  # reprocess even already-done templates
"""

import argparse
import io
import logging
from pathlib import Path
from typing import Any

import manifest as mf
from analyzer import analyze_thumbnail

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _has_subject(image_path: Path) -> bool:
    """
    Quick rembg check: return True if rembg finds a meaningful foreground subject.
    Prevents wasting time generating thumbnails for pure-text or logo-only thumbnails.
    """
    try:
        import numpy as np
        from rembg import remove as rembg_remove
        from PIL import Image

        with open(image_path, "rb") as f:
            data = f.read()
        result = rembg_remove(data)
        out = Image.open(io.BytesIO(result)).convert("RGBA")
        alpha = np.array(out)[:, :, 3]
        coverage = float((alpha > 127).sum()) / alpha.size
        return coverage > 0.04  # at least 4 % of pixels are foreground
    except Exception as exc:
        logger.warning("Subject check failed for %s: %s", image_path, exc)
        return True  # optimistic: don't discard on error


def process_template(entry: dict[str, Any], force: bool = False) -> bool:
    """Analyze a single template and store DNA in its manifest entry."""
    original_path = Path(entry["files"]["original"])

    if not original_path.exists():
        logger.warning("Original not found: %s", original_path)
        return False

    already_done = bool(entry.get("dna"))
    if already_done and not force:
        logger.debug("Already processed: %s", entry["template_id"])
        return True

    logger.info("Processing %s …", entry["template_id"])

    if not _has_subject(original_path):
        logger.warning("  No subject detected — skipping %s", entry["template_id"])
        entry["dna"] = None
        entry["has_subject"] = False
        return False

    dna = analyze_thumbnail(original_path)
    if not dna:
        logger.warning("  Analysis failed for %s", entry["template_id"])
        return False

    entry["dna"] = dna
    entry["has_subject"] = True

    logger.info(
        "  ✓ emotion=%-12s layout=%-30s contrast=%s",
        dna["emotion"], dna["layout"], dna["contrast_style"],
    )
    return True


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
            manifest_data["templates"][entry["template_id"]] = entry
            ok += 1
        else:
            fail += 1

    mf.save(manifest_data)
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
