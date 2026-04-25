# ViralThumb – Dynamic Thumbnail Template Engine

A "Thumbnail-as-a-Service" engine that maintains a self-refreshing library of
100+ viral YouTube thumbnail templates and lets you drop your own face into any
of them in seconds.

---

## Architecture

```
curator.py     →  Scout YouTube for viral thumbnails, download them, deduplicate
processor.py   →  Segment the person out of each thumbnail (mask + inpaint)
generator.py   →  Composite a user photo into any pre-processed template
manifest.json  →  Single source of truth for every template's metadata
```

### Directory layout

```
ViralThumb/
├── templates/
│   ├── gaming/          # active templates per niche
│   ├── tech/
│   ├── commentary/
│   ├── lifestyle/
│   ├── finance/
│   ├── fitness/
│   ├── food/
│   └── travel/
├── archive/             # pruned templates (auto-managed by janitor)
├── output/              # generated thumbnails
├── user_assets/         # place user photos here
├── models/              # SAM 2 checkpoints (optional)
├── curator.py
├── processor.py
├── generator.py
├── manifest.py
├── config.py
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For GPU-accelerated SAM 2 (optional):
```bash
git clone https://github.com/facebookresearch/segment-anything-2
cd segment-anything-2 && pip install -e .
# Download a checkpoint into models/
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set YOUTUBE_API_KEY
```

### 3. Run the curator (fetch trending thumbnails)

```bash
python curator.py                    # all niches
python curator.py --niche gaming     # one niche only
python curator.py --prune-only       # run janitor without fetching new ones
```

### 4. Process templates (generate masks + backgrounds)

```bash
python processor.py                  # all unprocessed templates
python processor.py --force          # reprocess everything
python processor.py --template gaming_abc123
```

### 5. Generate a thumbnail

```bash
python generator.py gaming_abc123 user_assets/my_photo.jpg
python generator.py gaming_abc123 user_assets/my_photo.jpg --mode insightface
python generator.py gaming_abc123 user_assets/my_photo.jpg --output output/final.jpg
```

#### Python API

```python
from generator import generate_from_template

out_path = generate_from_template(
    template_id="gaming_abc123",
    user_image_path="user_assets/my_photo.jpg",
    mode="alpha_blend",          # or "insightface"
)
print(f"Saved to {out_path}")
```

#### Query the manifest

```python
import manifest as mf

data = mf.load()

# Top 10 hottest gaming templates this week
top = mf.top_by_hotness(data, niche="gaming", n=10)
for t in top:
    print(t["template_id"], t["hotness_score"])
```

---

## Auto-Refresh (Cron / Task Scheduler)

### Linux / macOS – crontab

Run a full refresh every Sunday at 3 AM:

```cron
0 3 * * 0  cd /path/to/ViralThumb && .venv/bin/python curator.py >> logs/curator.log 2>&1
5 3 * * 0  cd /path/to/ViralThumb && .venv/bin/python processor.py >> logs/processor.log 2>&1
```

Edit your crontab with `crontab -e`.

### Windows – Task Scheduler

1. Open **Task Scheduler** → *Create Basic Task*.
2. Trigger: **Weekly**, Sunday, 03:00.
3. Action: **Start a program** → `python.exe`
   Arguments: `C:\path\to\ViralThumb\curator.py`
   Start in: `C:\path\to\ViralThumb`
4. Repeat for `processor.py` with a 5-minute delay.

### Recommended refresh interval

| Interval | Best for |
|----------|----------|
| **Weekly** (recommended) | Standard YouTube trend cycle |
| Daily | Fast-moving niches (gaming, news commentary) |
| Monthly | Evergreen niches (finance, fitness) |

---

## Configuration reference

All settings live in `config.py` and can be overridden via environment variables
(`.env` file is auto-loaded at runtime by `python-dotenv`).

| Variable | Default | Description |
|---|---|---|
| `YOUTUBE_API_KEY` | — | **Required.** YouTube Data API v3 key |
| `YOUTUBE_REGION_CODE` | `US` | Trending region |
| `SEGMENTATION_BACKEND` | `mediapipe` | `mediapipe` or `sam2` |
| `SAM2_CHECKPOINT` | `models/sam2_hiera_large.pt` | Path to SAM 2 weights |
| `COMPOSITING_MODE` | `alpha_blend` | `alpha_blend` or `insightface` |
| `INSIGHTFACE_MODEL_PACK` | `buffalo_l` | InsightFace model pack |
| `LOG_LEVEL` | `INFO` | Python log level |

Key constants in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_RESULTS_PER_NICHE` | 50 | Thumbnails fetched per niche per run |
| `MIN_VIEW_SUBSCRIBER_RATIO` | 0.10 | Minimum v/s ratio to qualify as viral |
| `MAX_TEMPLATE_AGE_DAYS` | 30 | Age after which templates are archived |
| `TARGET_LIBRARY_SIZE` | 100 | Cap on total active templates |
| `PHASH_DISTANCE_THRESHOLD` | 8 | Perceptual hash distance for dedup |

---

## How deduplication works

Every downloaded thumbnail is hashed with **pHash** (perceptual hash).
Before adding a new template, its hash is compared against all existing active
templates in the same niche. If the Hamming distance is ≤ `PHASH_DISTANCE_THRESHOLD`
(default 8 bits out of 64), the image is considered a visual duplicate and
discarded. This prevents the library from filling up with near-identical
"reaction face" thumbnails.

---

## Manifest structure

`manifest.json` is the engine's database. Each entry looks like:

```json
{
  "template_id": "gaming_abc12345",
  "niche": "gaming",
  "video_title": "I Survived 100 Days…",
  "view_count": 4200000,
  "subscriber_count": 800000,
  "hotness_score": 5.25,
  "status": "active",
  "added_at": "2026-04-25T03:00:00Z",
  "files": {
    "original": "templates/gaming/gaming_abc12345_original.jpg",
    "bg_only":  "templates/gaming/gaming_abc12345_bg_only.png",
    "mask":     "templates/gaming/gaming_abc12345_mask.png"
  }
}
```
