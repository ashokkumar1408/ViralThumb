"""
app.py – ViralThumb Flask Web App

Routes:
  GET  /                    → template browser + generator UI
  GET  /api/templates       → JSON list of active templates
  POST /api/generate        → composite user photo onto a template
  GET  /thumb/<path>        → serve original thumbnail images
  GET  /output/<filename>   → serve generated result images
"""

import uuid
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort

import manifest as mf
from config import OUTPUT_DIR, TEMPLATES_DIR, BASE_DIR
from generator import generate_from_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = Flask(__name__, template_folder="web", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_DIR = BASE_DIR / "user_assets"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    data = mf.load()
    niches = sorted({e["niche"] for e in mf.list_active(data)})
    return render_template("index.html", niches=niches)


# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/api/templates")
def api_templates():
    niche = request.args.get("niche")
    data = mf.load()
    entries = mf.list_active(data, niche=niche)
    entries = sorted(entries, key=lambda e: e.get("hotness_score", 0), reverse=True)

    result = []
    for e in entries:
        original = Path(e["files"]["original"])
        result.append({
            "template_id":   e["template_id"],
            "niche":         e["niche"],
            "video_title":   e.get("video_title", ""),
            "hotness_score": round(e.get("hotness_score", 0), 2),
            "view_count":    e.get("view_count", 0),
            "ready":         Path(e["files"]["bg_only"]).exists() and Path(e["files"]["mask"]).exists(),
            "thumbnail_url": f"/thumb/{original.relative_to(BASE_DIR)}" if original.exists() else None,
        })
    return jsonify(result)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    template_id = request.form.get("template_id", "").strip()
    mode = request.form.get("mode", "alpha_blend")
    photo = request.files.get("photo")

    if not template_id:
        return jsonify({"error": "template_id is required"}), 400
    if not photo or not _allowed(photo.filename):
        return jsonify({"error": "Upload a jpg/png photo"}), 400

    # Save uploaded photo
    ext = photo.filename.rsplit(".", 1)[1].lower()
    user_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"
    photo.save(str(user_path))

    try:
        out_path = generate_from_template(
            template_id=template_id,
            user_image_path=user_path,
            output_path=OUTPUT_DIR / f"{template_id}_{uuid.uuid4().hex[:8]}.jpg",
            mode=mode,
        )
        return jsonify({"result_url": f"/output/{out_path.name}"})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        user_path.unlink(missing_ok=True)


# ── Static asset serving ───────────────────────────────────────────────────────

@app.route("/thumb/<path:rel_path>")
def serve_thumb(rel_path):
    full = BASE_DIR / rel_path
    if not full.exists() or not full.is_file():
        abort(404)
    return send_file(full)


@app.route("/output/<filename>")
def serve_output(filename):
    full = OUTPUT_DIR / filename
    if not full.exists():
        abort(404)
    return send_file(full)


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
