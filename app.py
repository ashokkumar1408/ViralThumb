"""
app.py  –  ViralThumb Flask Web App

Routes
    GET  /                             → main UI
    GET  /api/templates                → list templates with metadata
    GET  /api/templates/preview/<id>   → rendered background preview (JPEG 320×180)
    POST /api/process-photo            → rembg + 4 expression variants → base64 dict
    POST /api/generate                 → composite thumbnail → result URL
    GET  /output/<filename>            → serve generated results
"""

import base64
import io
import logging
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from config import BASE_DIR, OUTPUT_DIR, YOUTUBE_API_KEY
from expressions import generate_variants
from generator import combine_thumbnail
from thumbnail_engine import TEMPLATES, get_background, list_templates, pre_render_all, refresh_templates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="web", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

UPLOAD_DIR = BASE_DIR / "user_assets"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = {"jpg", "jpeg", "png", "webp"}

# In-memory variant cache: session_id → {variant_name: PIL.Image}
_VARIANT_CACHE: dict[str, dict] = {}


def _ok_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def _pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/templates")
def api_templates():
    return jsonify(list_templates())


@app.route("/api/templates/preview/<template_id>")
def api_template_preview(template_id):
    if template_id not in TEMPLATES:
        abort(404)
    bg    = get_background(template_id)
    thumb = bg.resize((320, 180))
    buf   = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/process-photo", methods=["POST"])
def api_process_photo():
    photo = request.files.get("photo")
    if not photo or not _ok_ext(photo.filename or ""):
        return jsonify({"error": "Upload a jpg / png / webp photo"}), 400

    ext        = (photo.filename or "file.jpg").rsplit(".", 1)[1].lower()
    session_id = uuid.uuid4().hex
    user_path  = UPLOAD_DIR / f"{session_id}.{ext}"
    photo.save(str(user_path))

    try:
        variants = generate_variants(user_path)
        b64      = {name: _pil_to_b64(img) for name, img in variants.items()}
        _VARIANT_CACHE[session_id] = variants
        return jsonify({"session_id": session_id, "variants": b64})
    except Exception as exc:
        logger.exception("process-photo failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        user_path.unlink(missing_ok=True)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data        = request.get_json(force=True) or {}
    session_id  = data.get("session_id", "")
    template_id = data.get("template_id", "")
    variant     = data.get("variant", "original")
    headline    = data.get("headline", "").strip()
    sub_text    = data.get("sub_text", "").strip()

    if session_id not in _VARIANT_CACHE:
        return jsonify({"error": "Session expired — please re-upload your photo"}), 400
    if template_id not in TEMPLATES:
        return jsonify({"error": f"Unknown template: {template_id}"}), 400

    subject = _VARIANT_CACHE[session_id].get(variant)
    if subject is None:
        return jsonify({"error": f"Unknown variant: {variant}"}), 400

    try:
        out = combine_thumbnail(
            template_id = template_id,
            subject     = subject,
            headline    = headline,
            sub_text    = sub_text,
            output_path = OUTPUT_DIR / f"{template_id}_{uuid.uuid4().hex[:8]}.jpg",
        )
        return jsonify({"result_url": f"/output/{out.name}"})
    except Exception as exc:
        logger.exception("generate failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/fetch-youtube", methods=["POST"])
def api_fetch_youtube():
    """Trigger curator to pull fresh trending thumbnails from YouTube."""
    if not YOUTUBE_API_KEY:
        return jsonify({"error": "YOUTUBE_API_KEY is not set in your .env file"}), 400
    try:
        import curator
        curator.run()
        count = refresh_templates()
        return jsonify({"count": count})
    except EnvironmentError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("fetch-youtube failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/output/<path:filename>")
def serve_output(filename):
    full = OUTPUT_DIR / Path(filename).name
    if not full.exists():
        abort(404)
    return send_file(full)


if __name__ == "__main__":
    pre_render_all()
    app.run(debug=True, port=5000)
