"""
app.py  –  ViralThumb Flask Web App

Routes
    GET  /                  → main UI
    GET  /api/styles        → list available style presets
    POST /api/generate      → generate thumbnail (style + photo + text)
    GET  /output/<filename> → serve generated results
"""

import uuid
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort

from config import OUTPUT_DIR, BASE_DIR
from generator import generate_thumbnail, get_styles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = Flask(__name__, template_folder="web", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

UPLOAD_DIR = BASE_DIR / "user_assets"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = {"jpg", "jpeg", "png", "webp"}


def _ok_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/styles")
def api_styles():
    return jsonify(get_styles())


@app.route("/api/generate", methods=["POST"])
def api_generate():
    style_id = request.form.get("style", "shock").strip()
    headline = request.form.get("headline", "").strip()
    sub_text = request.form.get("sub_text", "").strip()
    photo    = request.files.get("photo")

    if not photo or not _ok_ext(photo.filename):
        return jsonify({"error": "Upload a jpg / png photo"}), 400

    ext       = photo.filename.rsplit(".", 1)[1].lower()
    user_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"
    photo.save(str(user_path))

    try:
        out = generate_thumbnail(
            style_id        = style_id,
            user_image_path = user_path,
            headline        = headline,
            sub_text        = sub_text,
            output_path     = OUTPUT_DIR / f"{style_id}_{uuid.uuid4().hex[:8]}.jpg",
        )
        return jsonify({"result_url": f"/output/{out.name}"})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        user_path.unlink(missing_ok=True)


@app.route("/output/<filename>")
def serve_output(filename):
    full = OUTPUT_DIR / filename
    if not full.exists():
        abort(404)
    return send_file(full)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
