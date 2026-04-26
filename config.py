"""Central configuration for the Thumbnail Template Engine."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env before any os.environ.get() calls below

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
ARCHIVE_DIR = BASE_DIR / "archive"
OUTPUT_DIR = BASE_DIR / "output"
USER_ASSETS_DIR = BASE_DIR / "user_assets"
MANIFEST_PATH = BASE_DIR / "manifest.json"

# ── YouTube API ────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
YOUTUBE_REGION_CODE = os.environ.get("YOUTUBE_REGION_CODE", "US")

# Niches to track. Each key is the folder name; value is the YouTube category ID.
# https://developers.google.com/youtube/v3/docs/videoCategories
NICHE_CATEGORY_MAP: dict[str, str] = {
    "gaming": "20",
    "tech": "28",
    "commentary": "22",   # People & Blogs (closest for commentary)
    "lifestyle": "26",
    "finance": "22",      # People & Blogs (finance channels use this)
    "fitness": "17",
    "food": "26",
    "travel": "19",
}

# ── Curation Settings ──────────────────────────────────────────────────────────
MAX_RESULTS_PER_NICHE = 50        # thumbnails fetched per niche per run
MIN_VIEW_SUBSCRIBER_RATIO = 0.10  # 10 % of subs as minimum viral signal
MAX_TEMPLATE_AGE_DAYS = 30        # archive templates older than this
TARGET_LIBRARY_SIZE = 100         # maximum live templates per niche combined
PHASH_DISTANCE_THRESHOLD = 8      # perceptual hash bits; lower = stricter dedup

# ── Image Processor ────────────────────────────────────────────────────────────
# Choose 'rembg' (default, CPU, no setup), 'mediapipe', or 'sam2' (GPU)
SEGMENTATION_BACKEND = os.environ.get("SEGMENTATION_BACKEND", "rembg")

# SAM 2 checkpoint (only used when SEGMENTATION_BACKEND='sam2')
SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    str(BASE_DIR / "models" / "sam2_hiera_large.pt"),
)
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "sam2_hiera_l.yaml")

# InsightFace model pack used by generator.py
INSIGHTFACE_MODEL_PACK = os.environ.get("INSIGHTFACE_MODEL_PACK", "buffalo_l")

# Output thumbnail resolution
THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
