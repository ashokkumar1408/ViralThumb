"""
curator.py – YouTube Scout + Template Repository Manager

Responsibilities:
  1. Query YouTube Data API for viral videos per niche.
  2. Download highest-res thumbnails + metadata.
  3. Deduplicate via perceptual hashing (pHash).
  4. Prune stale / low-hotness templates (the "Janitor").
  5. Keep manifest.json in sync.

Usage:
    python curator.py                    # run a full refresh cycle
    python curator.py --niche gaming     # refresh one niche only
    python curator.py --prune-only       # only run the janitor
"""

import argparse
import hashlib
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import imagehash
import requests
from googleapiclient.discovery import build
from PIL import Image

import manifest as mf
from analyzer import analyze_thumbnail
from config import (
    ARCHIVE_DIR,
    MAX_RESULTS_PER_NICHE,
    MAX_TEMPLATE_AGE_DAYS,
    NICHE_CATEGORY_MAP,
    PHASH_DISTANCE_THRESHOLD,
    TARGET_LIBRARY_SIZE,
    TEMPLATES_DIR,
    YOUTUBE_API_KEY,
    YOUTUBE_REGION_CODE,
    MIN_VIEW_SUBSCRIBER_RATIO,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── YouTube helpers ────────────────────────────────────────────────────────────

def _youtube_client():
    if not YOUTUBE_API_KEY:
        raise EnvironmentError("YOUTUBE_API_KEY is not set.")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def fetch_trending_videos(
    youtube,
    category_id: str,
    max_results: int = MAX_RESULTS_PER_NICHE,
) -> list[dict[str, Any]]:
    """Return video resource dicts from the mostPopular chart."""
    response = (
        youtube.videos()
        .list(
            part="snippet,statistics,contentDetails",
            chart="mostPopular",
            videoCategoryId=category_id,
            regionCode=YOUTUBE_REGION_CODE,
            maxResults=min(max_results, 50),
        )
        .execute()
    )
    return response.get("items", [])


def fetch_channel_subscriber_count(youtube, channel_id: str) -> int:
    resp = (
        youtube.channels()
        .list(part="statistics", id=channel_id)
        .execute()
    )
    items = resp.get("items", [])
    if not items:
        return 0
    return int(items[0]["statistics"].get("subscriberCount", 0))


def _viral_score(view_count: int, subscriber_count: int) -> float:
    """view-to-subscriber ratio; higher means more viral for the channel size."""
    if subscriber_count == 0:
        return 0.0
    return view_count / subscriber_count


def _best_thumbnail_url(snippet: dict) -> str:
    """Pick the highest resolution thumbnail available."""
    thumbs = snippet.get("thumbnails", {})
    for quality in ("maxres", "standard", "high", "medium", "default"):
        if quality in thumbs:
            return thumbs[quality]["url"]
    return ""


# ── Download helper ────────────────────────────────────────────────────────────

def download_image(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except requests.RequestException as exc:
            wait = 2 ** attempt
            logger.warning("Download failed (%s), retry in %ds: %s", exc, wait, url)
            time.sleep(wait)
    return False


# ── Perceptual hashing / dedup ─────────────────────────────────────────────────

def compute_phash(image_path: Path) -> str:
    img = Image.open(image_path)
    return str(imagehash.phash(img))


def is_duplicate(new_phash: str, existing_phashes: list[str]) -> bool:
    new_h = imagehash.hex_to_hash(new_phash)
    for ph in existing_phashes:
        if abs(new_h - imagehash.hex_to_hash(ph)) <= PHASH_DISTANCE_THRESHOLD:
            return True
    return False


# ── Template ID generation ─────────────────────────────────────────────────────

def make_template_id(niche: str, video_id: str) -> str:
    short = hashlib.md5(video_id.encode()).hexdigest()[:8]
    return f"{niche}_{short}"


# ── Janitor: prune stale templates ────────────────────────────────────────────

def prune_stale_templates(manifest_data: dict[str, Any]) -> int:
    """Archive templates older than MAX_TEMPLATE_AGE_DAYS. Returns count pruned."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_TEMPLATE_AGE_DAYS)
    pruned = 0
    for tid, entry in list(manifest_data["templates"].items()):
        if entry["status"] != "active":
            continue
        added = datetime.fromisoformat(entry["added_at"].replace("Z", "+00:00"))
        if added < cutoff:
            _archive_template(entry)
            mf.archive_entry(manifest_data, tid)
            pruned += 1
            logger.info("Archived stale template: %s", tid)
    return pruned


def _archive_template(entry: dict[str, Any]) -> None:
    for key, rel_path in entry.get("files", {}).items():
        src = Path(rel_path)
        if src.exists():
            dest = ARCHIVE_DIR / src.name
            shutil.move(str(src), str(dest))


# ── Core scout logic ───────────────────────────────────────────────────────────

def scout_niche(
    youtube,
    niche: str,
    category_id: str,
    manifest_data: dict[str, Any],
) -> int:
    """Fetch + download viral thumbnails for a single niche. Returns new count."""
    niche_dir = TEMPLATES_DIR / niche
    niche_dir.mkdir(parents=True, exist_ok=True)

    existing_phashes = [
        e["phash"]
        for e in mf.list_active(manifest_data, niche)
        if "phash" in e
    ]

    logger.info("Scouting niche '%s' (category %s)…", niche, category_id)
    videos = fetch_trending_videos(youtube, category_id, MAX_RESULTS_PER_NICHE)
    added = 0

    for video in videos:
        video_id = video["id"]
        snippet = video["snippet"]
        stats = video.get("statistics", {})
        channel_id = snippet.get("channelId", "")

        view_count = int(stats.get("viewCount", 0))
        sub_count = fetch_channel_subscriber_count(youtube, channel_id)
        ratio = _viral_score(view_count, sub_count)

        if ratio < MIN_VIEW_SUBSCRIBER_RATIO:
            logger.debug("Skipping %s — low v/s ratio %.2f", video_id, ratio)
            continue

        template_id = make_template_id(niche, video_id)
        if mf.get_entry(manifest_data, template_id):
            logger.debug("Template %s already exists, skipping.", template_id)
            continue

        thumb_url = _best_thumbnail_url(snippet)
        if not thumb_url:
            continue

        tmp_path = niche_dir / f"{template_id}_original.jpg"
        if not download_image(thumb_url, tmp_path):
            logger.warning("Could not download thumbnail for %s", video_id)
            continue

        phash = compute_phash(tmp_path)
        if is_duplicate(phash, existing_phashes):
            logger.debug("Duplicate detected for %s, removing.", template_id)
            tmp_path.unlink()
            continue

        existing_phashes.append(phash)

        entry: dict[str, Any] = {
            "template_id": template_id,
            "niche": niche,
            "original_url": thumb_url,
            "video_id": video_id,
            "video_title": snippet.get("title", ""),
            "channel_id": channel_id,
            "view_count": view_count,
            "subscriber_count": sub_count,
            "view_sub_ratio": round(ratio, 4),
            "phash": phash,
            "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_checked": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hotness_score": round(ratio, 4),
            "status": "active",
            "files": {
                "original": str(tmp_path),
            },
            "dna": analyze_thumbnail(tmp_path),
        }

        mf.add_entry(manifest_data, entry)
        added += 1
        logger.info(
            "Added %s (ratio=%.2f emotion=%s)",
            template_id, ratio, entry["dna"].get("emotion", "?"),
        )

        if mf.count_active(manifest_data) >= TARGET_LIBRARY_SIZE:
            logger.info("Library size cap reached (%d).", TARGET_LIBRARY_SIZE)
            break

    return added


# ── Entry point ────────────────────────────────────────────────────────────────

def run(niche_filter: str | None = None, prune_only: bool = False) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_data = mf.load()

    pruned = prune_stale_templates(manifest_data)
    logger.info("Janitor pruned %d stale templates.", pruned)

    if prune_only:
        mf.save(manifest_data)
        return

    youtube = _youtube_client()
    niche_map = (
        {niche_filter: NICHE_CATEGORY_MAP[niche_filter]}
        if niche_filter
        else NICHE_CATEGORY_MAP
    )

    total_added = 0
    for niche, cat_id in niche_map.items():
        total_added += scout_niche(youtube, niche, cat_id, manifest_data)

    mf.save(manifest_data)
    logger.info(
        "Curation complete. Added %d new templates; library now has %d active.",
        total_added,
        mf.count_active(manifest_data),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViralThumb curator")
    parser.add_argument("--niche", help="Refresh a single niche only")
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only run the janitor, skip scouting",
    )
    args = parser.parse_args()
    run(niche_filter=args.niche, prune_only=args.prune_only)
