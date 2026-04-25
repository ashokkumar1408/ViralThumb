"""
Manifest CRUD helpers.

The manifest.json schema per entry:
{
  "template_id": "gaming_abc123",
  "niche": "gaming",
  "original_url": "https://...",
  "video_id": "dQw4w9WgXcQ",
  "video_title": "...",
  "channel_id": "UC...",
  "view_count": 4200000,
  "subscriber_count": 800000,
  "view_sub_ratio": 5.25,
  "phash": "f8e0c0a0b0d0e0f0",
  "added_at": "2026-04-25T12:00:00Z",
  "last_checked": "2026-04-25T12:00:00Z",
  "hotness_score": 5.25,
  "status": "active",          // active | archived
  "files": {
    "original": "templates/gaming/gaming_abc123_original.jpg",
    "bg_only":  "templates/gaming/gaming_abc123_bg_only.png",
    "mask":     "templates/gaming/gaming_abc123_mask.png"
  }
}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MANIFEST_PATH

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"updated_at": _now_iso(), "templates": {}}


def save(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now_iso()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.debug("Manifest saved (%d templates)", len(manifest["templates"]))


def add_entry(manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    manifest["templates"][entry["template_id"]] = entry


def get_entry(manifest: dict[str, Any], template_id: str) -> dict[str, Any] | None:
    return manifest["templates"].get(template_id)


def list_active(manifest: dict[str, Any], niche: str | None = None) -> list[dict[str, Any]]:
    entries = [e for e in manifest["templates"].values() if e["status"] == "active"]
    if niche:
        entries = [e for e in entries if e["niche"] == niche]
    return entries


def top_by_hotness(
    manifest: dict[str, Any], niche: str | None = None, n: int = 10
) -> list[dict[str, Any]]:
    entries = list_active(manifest, niche)
    return sorted(entries, key=lambda e: e.get("hotness_score", 0), reverse=True)[:n]


def archive_entry(manifest: dict[str, Any], template_id: str) -> None:
    entry = manifest["templates"].get(template_id)
    if entry:
        entry["status"] = "archived"
        entry["archived_at"] = _now_iso()


def count_active(manifest: dict[str, Any], niche: str | None = None) -> int:
    return len(list_active(manifest, niche))
