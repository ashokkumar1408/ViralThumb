"""
thumbnail_engine.py – Load real YouTube thumbnails from the manifest as compositing templates.

Each template is a real viral YouTube thumbnail. When the user composites their
photo, we place their RGBA cutout over the original subject's position (derived
from the analyzer's subject_bbox DNA), so they literally replace the YouTuber.
"""

from pathlib import Path
from typing import Any

import manifest as mf
from PIL import Image

from config import THUMBNAIL_HEIGHT as H, THUMBNAIL_WIDTH as W


# ── Zone helpers ───────────────────────────────────────────────────────────────

def _face_zone(subject_bbox: dict | None) -> dict:
    """
    Convert the analyzer's normalized face rect into a full-person pixel zone.
    The face is ~25-35 % of a person's height in YouTube thumbnails, so we
    extrapolate downward to get a body-sized zone.
    """
    if not subject_bbox:
        # No face detected → use right half as default subject area
        return {"x1": W // 2, "y1": 0, "x2": W, "y2": H}

    fx  = int(subject_bbox["x"]  * W)
    fy  = int(subject_bbox["y"]  * H)
    fw  = int(subject_bbox["w"]  * W)
    fh  = int(subject_bbox["h"]  * H)
    fcx = int(subject_bbox["cx"] * W)

    # Face occupies roughly the top 30 % of the person; extrapolate body height
    person_h = min(H, int(fh / 0.30))
    person_w = max(int(fw * 1.8), int(W * 0.28))

    x1 = max(0, fcx - person_w // 2)
    x2 = min(W, fcx + person_w // 2)
    y1 = max(0, fy - int(fh * 0.15))       # small margin above forehead
    y2 = min(H, y1 + person_h)

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _text_zone(face_zone: dict, text_side: str) -> dict:
    pad = 28
    y1  = max(H // 8, pad)
    y2  = min(7 * H // 8, H - pad)

    if text_side == "left":
        x1 = pad
        x2 = max(pad + 80, face_zone["x1"] - pad)
    else:
        x1 = min(W - pad - 80, face_zone["x2"] + pad)
        x2 = W - pad

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


_EMOTION_COLORS: dict[str, tuple] = {
    "shock":      ((255, 235,   0), (255,  60,   0)),
    "excitement": ((255, 210,   0), (255, 120,   0)),
    "fear":       ((120, 200, 255), ( 20,  80, 200)),
    "curiosity":  ((  0, 220, 255), (  0,  80, 200)),
}

def _emotion_colors(emotion: str) -> tuple:
    return _EMOTION_COLORS.get(emotion, _EMOTION_COLORS["excitement"])


# ── Manifest loader ────────────────────────────────────────────────────────────

def _load_from_manifest() -> dict[str, dict]:
    manifest_data = mf.load()
    out: dict[str, dict] = {}

    for entry in mf.list_active(manifest_data):
        tid  = entry["template_id"]
        path = Path(entry["files"]["original"])
        if not path.exists():
            continue

        dna          = entry.get("dna", {})
        bbox         = dna.get("subject_bbox")
        text_side    = dna.get("text_side", "right")
        subject_side = dna.get("subject_side", "right")
        emotion      = dna.get("emotion", "excitement")

        fzone        = _face_zone(bbox)
        tzone        = _text_zone(fzone, text_side)
        oc, gc       = _emotion_colors(emotion)

        dominant = dna.get("dominant_colors", [[20, 20, 20], [180, 180, 180], [255, 120, 0]])
        preview  = [f"rgb({c[0]},{c[1]},{c[2]})" for c in dominant[:3]]

        title = entry.get("video_title", tid)
        name  = (title[:28] + "…") if len(title) > 30 else title

        out[tid] = {
            "name":           name,
            "niche":          entry.get("niche", ""),
            "face_side":      subject_side,
            "face_zone":      fzone,
            "text_zone":      tzone,
            "text_align":     text_side,
            "text_color":     (255, 255, 255),
            "text_stroke":    (0, 0, 0),
            "outline_color":  oc,
            "glow_color":     gc,
            "preview_colors": preview,
            "image_path":     str(path),
        }

    return out


# ── Module-level template dict (refreshable at runtime) ───────────────────────

TEMPLATES: dict[str, dict] = _load_from_manifest()


# ── Public API ─────────────────────────────────────────────────────────────────

def refresh_templates() -> int:
    """Reload TEMPLATES from manifest after curator has fetched new thumbnails."""
    global TEMPLATES
    TEMPLATES = _load_from_manifest()
    return len(TEMPLATES)


def get_background(template_id: str) -> Image.Image:
    tmpl = TEMPLATES.get(template_id)
    if not tmpl:
        raise ValueError(f"Unknown template: {template_id}")
    return Image.open(tmpl["image_path"]).convert("RGB").resize((W, H), Image.LANCZOS)


def list_templates() -> list[dict]:
    return [
        {
            "id":        tid,
            "name":      t["name"],
            "niche":     t["niche"],
            "colors":    t["preview_colors"],
            "face_side": t["face_side"],
        }
        for tid, t in TEMPLATES.items()
    ]


def pre_render_all() -> None:
    """No-op: real YouTube thumbnails need no pre-rendering."""
    pass
