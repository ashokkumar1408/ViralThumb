"""
demo.py – End-to-end pipeline test using synthetic assets.

Creates:
  1. A viral-style YouTube thumbnail (1280×720, bold text + coloured bg).
  2. A stylised AI-avatar photo (drawn with PIL).
Runs:
  • MediaPipe person segmentation  → mask.png
  • OpenCV inpainting              → bg_only.png
  • Alpha-blend compositing        → output/demo_result.jpg

No YouTube API key or real photos needed.
"""

import sys
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
TEST_DIR = BASE / "demo_assets"
TEST_DIR.mkdir(exist_ok=True)
(BASE / "output").mkdir(exist_ok=True)

THUMB_PATH   = TEST_DIR / "viral_thumbnail.jpg"
AVATAR_PATH  = TEST_DIR / "ai_avatar.png"
MASK_PATH    = TEST_DIR / "mask.png"
BGONLY_PATH  = TEST_DIR / "bg_only.png"
OUTPUT_PATH  = BASE / "output" / "demo_result.jpg"

W, H = 1280, 720


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BUILD A VIRAL-STYLE THUMBNAIL
# ══════════════════════════════════════════════════════════════════════════════

def make_thumbnail() -> None:
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    # Gradient background: deep blue → purple
    for y in range(H):
        t = y / H
        r = int(8   + t * 30)
        g = int(8   + t * 10)
        b = int(120 + t * 80)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Neon-yellow "shock panel" on the left
    panel_w = int(W * 0.48)
    for x in range(panel_w):
        t = x / panel_w
        r = int(255 - t * 30)
        g = int(210 - t * 60)
        b = int(0)
        draw.line([(x, 0), (x, H)], fill=(r, g, b))

    # ── Draw a stylised "shocked presenter" silhouette ──────────────────────
    # Head
    head_cx, head_cy, head_r = 320, 220, 110
    draw.ellipse(
        [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
        fill=(220, 170, 120),
    )
    # Neck
    draw.rectangle([head_cx - 40, head_cy + head_r - 10, head_cx + 40, head_cy + head_r + 60],
                   fill=(210, 160, 110))
    # Torso
    draw.polygon(
        [(head_cx - 160, H), (head_cx + 160, H),
         (head_cx + 100, head_cy + head_r + 55), (head_cx - 100, head_cy + head_r + 55)],
        fill=(20, 80, 160),
    )
    # Wide-open shocked mouth
    draw.ellipse([head_cx - 35, head_cy + 20, head_cx + 35, head_cy + 80],
                 fill=(40, 20, 10))
    draw.ellipse([head_cx - 28, head_cy + 28, head_cx + 28, head_cy + 72],
                 fill=(180, 60, 60))
    # Raised eyebrows
    draw.arc([head_cx - 70, head_cy - 70, head_cx - 20, head_cy - 30],
             start=200, end=340, fill=(80, 50, 20), width=6)
    draw.arc([head_cx + 20, head_cy - 70, head_cx + 70, head_cy - 30],
             start=200, end=340, fill=(80, 50, 20), width=6)
    # Eyes (wide open)
    draw.ellipse([head_cx - 60, head_cy - 30, head_cx - 25, head_cy + 5],
                 fill=(255, 255, 255))
    draw.ellipse([head_cx + 25, head_cy - 30, head_cx + 60, head_cy + 5],
                 fill=(255, 255, 255))
    draw.ellipse([head_cx - 50, head_cy - 22, head_cx - 32, head_cy - 4],
                 fill=(50, 80, 160))
    draw.ellipse([head_cx + 32, head_cy - 22, head_cx + 50, head_cy - 4],
                 fill=(50, 80, 160))
    # Hands thrown up
    draw.ellipse([head_cx - 220, head_cy + 50, head_cx - 150, head_cy + 120],
                 fill=(220, 170, 120))
    draw.ellipse([head_cx + 150, head_cy + 50, head_cx + 220, head_cy + 120],
                 fill=(220, 170, 120))
    # Arms
    draw.line([head_cx - 100, head_cy + head_r + 80, head_cx - 185, head_cy + 85],
              fill=(20, 80, 160), width=40)
    draw.line([head_cx + 100, head_cy + head_r + 80, head_cx + 185, head_cy + 85],
              fill=(20, 80, 160), width=40)

    # ── Text on the right side ──────────────────────────────────────────────
    try:
        font_big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        font_mid  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 55)
        font_sm   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
    except OSError:
        font_big = font_mid = font_sm = ImageFont.load_default()

    tx = panel_w + 30

    # "I CAN'T BELIEVE"
    draw.text((tx, 60),  "I CAN'T",   fill=(255, 230, 0),  font=font_big)
    draw.text((tx, 155), "BELIEVE",   fill=(255, 255, 255), font=font_big)

    # Red pill badge
    draw.rounded_rectangle([tx, 270, tx + 440, 355], radius=20, fill=(220, 30, 30))
    draw.text((tx + 18, 278), "THIS ACTUALLY WORKS", fill=(255, 255, 255), font=font_sm)

    # View count badge
    draw.rounded_rectangle([tx, 375, tx + 320, 445], radius=15, fill=(0, 0, 0, 200))
    draw.text((tx + 16, 385), "47M VIEWS", fill=(255, 215, 0), font=font_mid)

    # Arrow pointing at presenter
    for i in range(5):
        ax = tx - 10 - i * 22
        draw.polygon(
            [(ax, H // 2 - 18), (ax - 30, H // 2 - 40), (ax - 30, H // 2 + 40), (ax, H // 2 + 18)],
            fill=(255, 50, 50),
        )

    # Subtle vignette
    vignette = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vignette)
    for i in range(120):
        alpha = int(i * 1.5)
        vd.rectangle([i, i, W - i, H - i], outline=alpha)
    img = Image.composite(img, Image.new("RGB", (W, H), (0, 0, 0)),
                          ImageOps_invert(vignette))

    img.save(THUMB_PATH, quality=95)
    print(f"[1/5] Thumbnail saved → {THUMB_PATH}")


def ImageOps_invert(img: Image.Image) -> Image.Image:
    from PIL import ImageOps
    return ImageOps.invert(img)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD AN AI AVATAR
# ══════════════════════════════════════════════════════════════════════════════

def make_avatar() -> None:
    """
    Draw a stylised AI/robot avatar: glowing circuit-face on dark background.
    """
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    draw.ellipse([0, 0, size, size], fill=(15, 15, 30, 255))

    # Outer glow ring
    for i in range(18, 0, -1):
        alpha = int(80 * (1 - i / 18))
        draw.ellipse([i * 1, i * 1, size - i, size - i],
                     outline=(0, 200, 255, alpha), width=3)

    # Face oval (metallic silver)
    fx1, fy1, fx2, fy2 = 80, 90, size - 80, size - 60
    draw.ellipse([fx1, fy1, fx2, fy2], fill=(60, 70, 90))
    draw.ellipse([fx1 + 4, fy1 + 4, fx2 - 4, fy2 - 4], fill=(75, 85, 105))

    cx = size // 2

    # Visor (glowing cyan)
    vx1, vy1, vx2, vy2 = 105, 155, size - 105, 270
    draw.rounded_rectangle([vx1, vy1, vx2, vy2], radius=30, fill=(0, 30, 50))
    # Glowing eyes inside visor
    for eye_cx in (cx - 60, cx + 60):
        for r in range(28, 8, -4):
            alpha = int(255 * ((28 - r) / 20))
            draw.ellipse([eye_cx - r, 195 - r, eye_cx + r, 195 + r],
                         fill=(0, min(200 + alpha, 255), 255))
        draw.ellipse([eye_cx - 10, 185, eye_cx + 10, 205], fill=(255, 255, 255))

    # Scan line across visor
    draw.line([(vx1 + 10, 213), (vx2 - 10, 213)], fill=(0, 255, 255, 80), width=2)

    # Nose plate
    draw.rounded_rectangle([cx - 18, 275, cx + 18, 310], radius=6, fill=(50, 60, 80))

    # Mouth / speaker grille
    draw.rounded_rectangle([cx - 65, 325, cx + 65, 370], radius=12, fill=(30, 35, 50))
    for i in range(-3, 4):
        draw.line([(cx + i * 18, 335), (cx + i * 18, 360)],
                  fill=(0, 200, 255), width=3)

    # Circuit lines on cheeks
    for sign, ox in [(1, cx + 90), (-1, cx - 90)]:
        draw.line([(ox, 200), (ox + sign * 30, 200)], fill=(0, 200, 255), width=2)
        draw.line([(ox + sign * 30, 200), (ox + sign * 30, 240)],
                  fill=(0, 200, 255), width=2)
        draw.ellipse([ox + sign * 25, 235, ox + sign * 35, 245],
                     fill=(0, 200, 255))

    # Antenna
    draw.line([(cx, 90), (cx, 40)], fill=(150, 160, 180), width=5)
    draw.ellipse([cx - 12, 28, cx + 12, 52], fill=(0, 200, 255))

    # Neck + shoulders
    draw.rectangle([cx - 30, fy2 - 10, cx + 30, fy2 + 40], fill=(60, 70, 90))
    draw.ellipse([cx - 130, fy2 + 10, cx + 130, fy2 + 100], fill=(55, 65, 85))

    img.save(AVATAR_PATH, format="PNG")
    print(f"[2/5] AI avatar saved → {AVATAR_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SEGMENT THE PRESENTER (MediaPipe)
# ══════════════════════════════════════════════════════════════════════════════

def segment_thumbnail() -> np.ndarray:
    """Run MediaPipe selfie segmentation and return the mask."""
    import mediapipe as mp

    img_bgr = cv2.imread(str(THUMB_PATH))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(img_rgb)

    raw_mask = result.segmentation_mask  # float32 H×W in [0,1]

    # Threshold + morphological cleanup
    mask = (raw_mask > 0.45).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # If MediaPipe found nothing (pure synthetic image), fall back to a
    # hand-crafted mask that covers the drawn presenter region.
    coverage = np.count_nonzero(mask) / mask.size
    if coverage < 0.02:
        print("   MediaPipe found no person — using synthetic shape mask as fallback.")
        mask = np.zeros((H, W), dtype=np.uint8)
        # Match the drawn presenter: head + torso region on the left panel
        head_cx, head_cy, head_r = 320, 220, 130
        cv2.ellipse(mask, (head_cx, head_cy), (head_r, head_r), 0, 0, 360, 255, -1)
        # Torso polygon
        pts = np.array([[160, H], [480, H], [420, head_cy + head_r + 55],
                        [220, head_cy + head_r + 55]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        # Arms
        cv2.ellipse(mask, (head_cx, head_cy + head_r + 80), (100, 40), 0, 0, 360, 255, -1)
        # Hands
        cv2.ellipse(mask, (head_cx - 185, head_cy + 85), (45, 40), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (head_cx + 185, head_cy + 85), (45, 40), 0, 0, 360, 255, -1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel2, iterations=2)

    cv2.imwrite(str(MASK_PATH), mask)
    print(f"[3/5] Mask saved    → {MASK_PATH}  (coverage {np.count_nonzero(mask)/mask.size*100:.1f}%)")
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 4.  INPAINT (remove presenter → background only)
# ══════════════════════════════════════════════════════════════════════════════

def inpaint_thumbnail(mask: np.ndarray) -> None:
    img_bgr = cv2.imread(str(THUMB_PATH))
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    result  = cv2.inpaint(img_bgr, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(str(BGONLY_PATH), result)
    print(f"[4/5] BG-only saved → {BGONLY_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  COMPOSITE AVATAR → BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════

def composite_avatar(mask: np.ndarray) -> None:
    bg_bgr  = cv2.imread(str(BGONLY_PATH))
    avatar  = cv2.imread(str(AVATAR_PATH), cv2.IMREAD_UNCHANGED)  # BGRA

    # ── Find mask bounding box ────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours in mask — aborting composite.")
        return
    rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # ── Resize avatar to fill the bounding box ────────────────────────────────
    av_resized = cv2.resize(avatar, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

    # ── Build smooth alpha channel ────────────────────────────────────────────
    # Use mask ROI for shape, AND avatar's own alpha for circular shape
    mask_roi = mask[ry:ry+rh, rx:rx+rw].astype(np.float32) / 255.0

    if av_resized.shape[2] == 4:
        av_alpha = av_resized[:, :, 3].astype(np.float32) / 255.0
        combined_alpha = mask_roi * av_alpha
        av_bgr = av_resized[:, :, :3]
    else:
        combined_alpha = mask_roi
        av_bgr = av_resized

    # Feather the edge
    combined_alpha = cv2.GaussianBlur(combined_alpha, (31, 31), 0)
    alpha_3ch = combined_alpha[:, :, np.newaxis]

    bg_roi   = bg_bgr[ry:ry+rh, rx:rx+rw].astype(np.float32)
    av_float = av_bgr.astype(np.float32)

    blended = (av_float * alpha_3ch + bg_roi * (1.0 - alpha_3ch)).clip(0, 255).astype(np.uint8)
    result  = bg_bgr.copy()
    result[ry:ry+rh, rx:rx+rw] = blended

    # Add a subtle glow ring around the avatar to make it pop
    glow_mask = (combined_alpha * 255).astype(np.uint8)
    glow_dilated = cv2.dilate(glow_mask, np.ones((15, 15), np.uint8), iterations=3)
    glow_edge = cv2.subtract(glow_dilated, glow_mask)
    glow_edge_f = glow_edge.astype(np.float32) / 255.0
    glow_color = np.full((rh, rw, 3), [255, 220, 0], dtype=np.float32)  # gold
    result_roi = result[ry:ry+rh, rx:rx+rw].astype(np.float32)
    glow_alpha_3ch = glow_edge_f[:, :, np.newaxis] * 0.6
    result[ry:ry+rh, rx:rx+rw] = (
        glow_color * glow_alpha_3ch + result_roi * (1.0 - glow_alpha_3ch)
    ).clip(0, 255).astype(np.uint8)

    cv2.imwrite(str(OUTPUT_PATH), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[5/5] Result saved  → {OUTPUT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== ViralThumb Pipeline Demo ===\n")
    make_thumbnail()
    make_avatar()
    mask = segment_thumbnail()
    inpaint_thumbnail(mask)
    composite_avatar(mask)
    print("\nDone! Open output/demo_result.jpg to see the composited thumbnail.")
    print("Intermediate files are in demo_assets/\n")
