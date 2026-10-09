"""Dataset Builder for Learned Photo-vs-Non-Photo Gate Classifier.

Generates balanced training and held-out validation sets:
1. Photos:
   - Natural camera photographs (portraits, selfies, indoor, outdoor, night, crowds, food)
   - WhatsApp-recompressed versions (JPEG Q=40..85, downscaled, portrait 720x1600 / 1080x1920)
   - Photorealistic AI generations (Midjourney, SDXL, CIFAKE)
2. Non-Photos:
   - Web application screenshots & mobile app UIs (Chrome headless + Pillow)
   - Academic / corporate certificates & diplomas
   - Business presentation slides
   - Formal documents, receipts, invoices, code editors
   - Statistical charts (bar, line, scatter, pie)
   - Vector logos, flat icons, posters, graphic designs
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_gate_dataset")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GATE_DIR = DATA_DIR / "gate_dataset"

TRAIN_PHOTO = GATE_DIR / "train" / "photo"
TRAIN_NON_PHOTO = GATE_DIR / "train" / "non_photo"
VAL_PHOTO = GATE_DIR / "val" / "photo"
VAL_NON_PHOTO = GATE_DIR / "val" / "non_photo"

CHROME_BIN = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def setup_dirs():
    for d in [TRAIN_PHOTO, TRAIN_NON_PHOTO, VAL_PHOTO, VAL_NON_PHOTO]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Non-Photo Generators: Web Pages, Docs, Charts, Slides, Certificates
# ---------------------------------------------------------------------------

def render_chrome_html(html: str, out_path: Path, width: int = 1280, height: int = 720):
    """Renders HTML string to PNG via Headless Chrome."""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        tmp_html = f.name
    try:
        cmd = [
            CHROME_BIN,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={out_path.resolve()}",
            f"file:///{Path(tmp_html).resolve().as_posix()}"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.warning(f"Chrome render error: {e}, falling back to PIL")
        img = Image.new("RGB", (width, height), (250, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), "HTML Render Fallback", fill=(20, 20, 20))
        img.save(out_path)
    finally:
        if os.path.exists(tmp_html):
            os.unlink(tmp_html)


def make_html_templates() -> List[Tuple[str, int, int]]:
    """Generates a collection of real web page and app UI HTML strings."""
    templates = []
    # 1. Code Editor / GitHub repo
    templates.append((
        """<!DOCTYPE html><html><body style="margin:0;padding:24px;background:#0d1117;color:#c9d1d9;font-family:Consolas,monospace;">
        <h2 style="color:#58a6ff;font-family:sans-serif;">src/detectors/image_type.py</h2>
        <div style="background:#161b22;padding:16px;border-radius:6px;border:1px solid #30363d;line-height:1.6;">
        <span style="color:#ff7b72;">import</span> torch<br>
        <span style="color:#ff7b72;">from</span> PIL <span style="color:#ff7b72;">import</span> Image<br>
        <span style="color:#d2a8ff;">def</span> <span style="color:#79c0ff;">classify_image</span>(img):<br>
        &nbsp;&nbsp;&nbsp;&nbsp;emb = model.extract(img)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#ff7b72;">return</span> classifier.predict(emb)<br>
        </div><div style="margin-top:20px;color:#8b949e;">Commit 4a8f9c · 14 files changed</div>
        </body></html>""", 1280, 720
    ))
    # 2. Wikipedia-style encyclopedia
    templates.append((
        """<!DOCTYPE html><html><body style="margin:0;padding:40px;background:#ffffff;color:#202122;font-family:sans-serif;line-height:1.6;">
        <h1 style="border-bottom:1px solid #a2a9b1;font-family:serif;font-size:28px;">Digital Image Forensics</h1>
        <p>From Wikipedia, the free encyclopedia</p>
        <p><b>Digital image forensics</b> is the technical discipline aimed at inferring the history of a digital image. In the absence of digital watermarks or signatures, blind methods rely solely on natural image statistics and physical sensor artifacts.</p>
        <div style="background:#f8f9fa;border:1px solid #c8ccd1;padding:12px;margin:20px 0;width:300px;">
        <b>Contents</b><ul><li>1 Principles</li><li>2 Compression fingerprints</li><li>3 Neural synthesis</li></ul>
        </div>
        </body></html>""", 1080, 800
    ))
    # 3. SaaS Analytics Dashboard
    templates.append((
        """<!DOCTYPE html><html><body style="margin:0;padding:20px;background:#f1f5f9;font-family:Inter,sans-serif;">
        <div style="display:flex;justify-content:space-between;background:#fff;padding:16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
        <h3 style="margin:0;color:#0f172a;">API Metrics & Cloud Gateways</h3>
        <span style="color:#10b981;font-weight:bold;">99.98% Healthy</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:20px;">
        <div style="background:#fff;padding:20px;border-radius:8px;border:1px solid #e2e8f0;">
        <div style="color:#64748b;font-size:12px;">ACTIVE SESSIONS</div>
        <div style="font-size:24px;font-weight:bold;margin-top:4px;">14,290</div>
        </div>
        <div style="background:#fff;padding:20px;border-radius:8px;border:1px solid #e2e8f0;">
        <div style="color:#64748b;font-size:12px;">AVG LATENCY</div>
        <div style="font-size:24px;font-weight:bold;margin-top:4px;color:#2563eb;">42 ms</div>
        </div>
        <div style="background:#fff;padding:20px;border-radius:8px;border:1px solid #e2e8f0;">
        <div style="color:#64748b;font-size:12px;">ERROR RATE</div>
        <div style="font-size:24px;font-weight:bold;margin-top:4px;color:#ef4444;">0.01%</div>
        </div>
        </div>
        </body></html>""", 1280, 720
    ))
    # 4. Mobile Chat / Messaging App Screenshot
    templates.append((
        """<!DOCTYPE html><html><body style="margin:0;padding:0;background:#e5ddd5;font-family:sans-serif;display:flex;flex-direction:column;height:100vh;">
        <div style="background:#075e54;color:#fff;padding:14px;font-weight:bold;display:flex;align-items:center;">
        <div style="width:36px;height:36px;background:#128c7e;border-radius:50%;margin-right:12px;"></div>
        <div>Alex Rivera<div style="font-size:11px;font-weight:normal;">online</div></div>
        </div>
        <div style="padding:16px;display:flex;flex-direction:column;gap:12px;">
        <div style="background:#fff;padding:10px 14px;border-radius:8px;max-width:75%;align-self:flex-start;box-shadow:0 1px 1px rgba(0,0,0,0.1);">
        Hey, did you review the forensic report?
        <div style="font-size:10px;color:#999;text-align:right;">10:14 AM</div>
        </div>
        <div style="background:#dcf8c6;padding:10px 14px;border-radius:8px;max-width:75%;align-self:flex-end;box-shadow:0 1px 1px rgba(0,0,0,0.1);">
        Yes, the image-type classifier is passing all tests now.
        <div style="font-size:10px;color:#999;text-align:right;">10:15 AM ✓✓</div>
        </div>
        </div>
        </body></html>""", 720, 1280
    ))
    # 5. Financial Invoice
    templates.append((
        """<!DOCTYPE html><html><body style="margin:0;padding:40px;background:#fff;color:#1e293b;font-family:sans-serif;">
        <div style="display:flex;justify-content:space-between;border-bottom:2px solid #e2e8f0;padding-bottom:16px;">
        <h2>INVOICE #INV-2026-881</h2>
        <div style="text-align:right;color:#64748b;">Date: Oct 8, 2026<br>Due: Net 30</div>
        </div>
        <table style="width:100%;margin-top:24px;border-collapse:collapse;" border="1" cellpadding="8">
        <tr style="background:#f8fafc;"><th>Description</th><th>Qty</th><th>Unit Price</th><th>Amount</th></tr>
        <tr><td>Security Audit & Forensic Validation</td><td>1</td><td>$2,400.00</td><td>$2,400.00</td></tr>
        <tr><td>Cloud Pipeline Deployment</td><td>2</td><td>$650.00</td><td>$1,300.00</td></tr>
        <tr style="font-weight:bold;"><td colspan="3" align="right">TOTAL DUE:</td><td>$3,700.00</td></tr>
        </table>
        <div style="margin-top:40px;color:#94a3b8;font-size:12px;">Payment instructions: Wire transfer within 30 days.</div>
        </body></html>""", 900, 1100
    ))
    return templates


def draw_pillow_chart(seed: int) -> Image.Image:
    """Draws statistical data charts via Pillow."""
    rng = random.Random(seed)
    w, h = 900, 600
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Chart frame
    draw.rectangle([10, 10, w - 10, h - 10], outline=(220, 224, 230), width=1)
    draw.text((40, 30), f"Statistical Analysis Series #{seed}", fill=(30, 41, 59))

    # Axes
    draw.line([100, 500, 800, 500], fill=(71, 85, 105), width=2)
    draw.line([100, 100, 100, 500], fill=(71, 85, 105), width=2)

    chart_type = rng.choice(["bar", "line", "scatter"])
    colors = [(59, 130, 246), (16, 185, 129), (239, 68, 68), (168, 85, 247), (245, 158, 11)]

    if chart_type == "bar":
        n_bars = rng.randint(5, 10)
        bar_w = 550 // n_bars
        for i in range(n_bars):
            val = rng.randint(40, 360)
            x0 = 130 + i * (bar_w + 10)
            y0 = 500 - val
            draw.rectangle([x0, y0, x0 + bar_w, 500], fill=colors[i % len(colors)], outline=(30, 41, 59))
    elif chart_type == "line":
        points = []
        for i in range(12):
            x = 120 + i * 55
            y = 500 - rng.randint(50, 380)
            points.append((x, y))
        draw.line(points, fill=colors[0], width=3)
        for pt in points:
            draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=colors[1], outline=(15, 23, 42))
    else:
        for _ in range(40):
            x = rng.randint(120, 780)
            y = rng.randint(120, 480)
            r = rng.randint(4, 9)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=colors[rng.randint(0, len(colors)-1)])

    return img


def draw_pillow_slide(seed: int) -> Image.Image:
    """Draws corporate presentation slides."""
    rng = random.Random(seed)
    w, h = 1024, 768
    bg_color = rng.choice([(255, 255, 255), (15, 23, 42), (248, 250, 252)])
    is_dark = (bg_color[0] < 50)
    text_color = (248, 250, 252) if is_dark else (15, 23, 42)
    accent = (59, 130, 246)

    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    # Top accent bar
    draw.rectangle([0, 0, w, 12], fill=accent)
    draw.text((60, 50), f"Q{rng.randint(1,4)} 2026 Strategy Review", fill=text_color)
    draw.text((60, 85), "Enterprise Security & Threat Detection", fill=(148, 163, 184))

    # Content boxes
    for i in range(3):
        x = 60 + i * 305
        box_bg = (30, 41, 59) if is_dark else (241, 245, 249)
        draw.rectangle([x, 150, x + 285, 650], fill=box_bg, outline=accent if i == 0 else None)
        draw.text((x + 20, 180), f"Pillar 0{i+1}", fill=accent)
        draw.text((x + 20, 220), f"Key Initiative {chr(65+i)}", fill=text_color)
        draw.text((x + 20, 260), "• Automated pipeline validation", fill=(148, 163, 184))
        draw.text((x + 20, 290), "• Real-time edge inference", fill=(148, 163, 184))
        draw.text((x + 20, 320), "• Multi-signal consensus", fill=(148, 163, 184))

    return img


def draw_pillow_poster_logo(seed: int) -> Image.Image:
    """Draws vector-style flat logos and geometric posters."""
    rng = random.Random(seed)
    w, h = 800, 800
    bg = rng.choice([(255, 255, 255), (243, 244, 246), (17, 24, 39)])
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2
    for r in range(250, 50, -40):
        color = (rng.randint(30, 240), rng.randint(30, 240), rng.randint(30, 240))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=rng.randint(2, 6))

    draw.polygon([(cx, cy - 120), (cx + 100, cy + 80), (cx - 100, cy + 80)], fill=(59, 130, 246))
    draw.text((cx - 80, cy + 180), "VECTOR LOGO DESIGN", fill=(100, 116, 139))
    return img


# ---------------------------------------------------------------------------
# 2. Photo Augmentations: WhatsApp Compression, Resizing, Noise, Selfies
# ---------------------------------------------------------------------------

def apply_whatsapp_compression(img: Image.Image, quality: int = 55, target_dim: Tuple[int, int] = None) -> Image.Image:
    """Simulates WhatsApp chat recompression and portrait dimension downscaling."""
    work = img.copy()
    if target_dim:
        work = work.resize(target_dim, Image.Resampling.LANCZOS)
    elif max(work.width, work.height) > 1600:
        scaling = 1600 / max(work.width, work.height)
        work = work.resize((int(work.width * scaling), int(work.height * scaling)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def create_mirror_selfie_simulation(source_img: Image.Image, seed: int) -> Image.Image:
    """Crops a photo into portrait orientation (720x1600 / 1080x1920) and simulates indoor mirror lighting."""
    rng = random.Random(seed)
    w, h = source_img.size

    # Crop to 9:16 or 9:20 portrait ratio
    target_aspect = rng.choice([9 / 16, 9 / 20, 9 / 19.5])
    if w / h > target_aspect:
        new_w = int(h * target_aspect)
        x0 = rng.randint(0, max(0, w - new_w))
        crop_box = (x0, 0, x0 + new_w, h)
    else:
        new_h = int(w / target_aspect)
        y0 = rng.randint(0, max(0, h - new_h))
        crop_box = (0, y0, w, y0 + new_h)

    cropped = source_img.crop(crop_box)
    target_res = rng.choice([(720, 1600), (1080, 1920), (720, 1280)])
    resized = cropped.resize(target_res, Image.Resampling.LANCZOS)

    # Slight indoor warm tint or contrast adjustment
    enh = ImageEnhance.Color(resized)
    tweaked = enh.enhance(rng.uniform(0.8, 1.2))
    return apply_whatsapp_compression(tweaked, quality=rng.randint(45, 75))


# ---------------------------------------------------------------------------
# 3. Main Builder Execution
# ---------------------------------------------------------------------------

def build_gate_dataset():
    setup_dirs()
    logger.info("Building Gate Classifier Dataset...")

    # A. Collect Source Real Photos
    source_real_dirs = [
        DATA_DIR / "tuning" / "real",
        DATA_DIR / "hard_tuning" / "real",
        DATA_DIR / "messy_real",
        DATA_DIR / "my_images" / "real",
    ]
    real_paths: List[Path] = []
    for d in source_real_dirs:
        if d.exists():
            real_paths.extend([p for p in d.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])

    # Collect Source AI Photos (photorealistic AI)
    source_ai_dirs = [
        DATA_DIR / "tuning" / "ai",
        DATA_DIR / "hard_tuning" / "ai",
    ]
    ai_paths: List[Path] = []
    for d in source_ai_dirs:
        if d.exists():
            ai_paths.extend([p for p in d.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])

    logger.info(f"Loaded {len(real_paths)} real photo sources, {len(ai_paths)} photorealistic AI sources.")

    # Split real and AI photos into train and val splits (75% train, 25% val)
    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(ai_paths)

    train_real = real_paths[: int(len(real_paths) * 0.75)]
    val_real = real_paths[int(len(real_paths) * 0.75) :]

    train_ai = ai_paths[: int(len(ai_paths) * 0.75)]
    val_ai = ai_paths[int(len(ai_paths) * 0.75) :]

    # 1. Process Train Photos
    p_idx = 0
    for p in train_real:
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                img_rgb.save(TRAIN_PHOTO / f"train_photo_real_{p_idx:04d}.jpg", quality=90)
                p_idx += 1

                # Add WhatsApp compressed version (JPEG Q=45..65)
                wa_img = apply_whatsapp_compression(img_rgb, quality=random.randint(45, 65))
                wa_img.save(TRAIN_PHOTO / f"train_photo_wa_{p_idx:04d}.jpg")
                p_idx += 1

                # Add portrait phone crop (mirror selfie style 720x1600)
                selfie_sim = create_mirror_selfie_simulation(img_rgb, seed=p_idx)
                selfie_sim.save(TRAIN_PHOTO / f"train_photo_selfie_{p_idx:04d}.jpg")
                p_idx += 1
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")

    for p in train_ai:
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                img_rgb.save(TRAIN_PHOTO / f"train_photo_ai_{p_idx:04d}.jpg", quality=85)
                p_idx += 1
                if random.random() < 0.5:
                    wa_ai = apply_whatsapp_compression(img_rgb, quality=random.randint(50, 75))
                    wa_ai.save(TRAIN_PHOTO / f"train_photo_ai_wa_{p_idx:04d}.jpg")
                    p_idx += 1
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")

    logger.info(f"Saved {p_idx} training photo samples to {TRAIN_PHOTO}")

    # 2. Process Val Photos
    v_idx = 0
    for p in val_real:
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                img_rgb.save(VAL_PHOTO / f"val_photo_real_{v_idx:04d}.jpg", quality=90)
                v_idx += 1
                # WhatsApp version
                wa_img = apply_whatsapp_compression(img_rgb, quality=55, target_dim=(720, 1600))
                wa_img.save(VAL_PHOTO / f"val_photo_wa_{v_idx:04d}.jpg")
                v_idx += 1
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")

    # Add user's real WhatsApp images to validation
    user_wa_dir = DATA_DIR / "my_images" / "real_whatsapp"
    if user_wa_dir.exists():
        for p in user_wa_dir.iterdir():
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(p) as img:
                        img.convert("RGB").save(VAL_PHOTO / f"val_user_whatsapp_{p.stem}.jpg")
                        v_idx += 1
                except Exception as e:
                    logger.warning(f"Error copying {p}: {e}")

    logger.info(f"Saved {v_idx} validation photo samples to {VAL_PHOTO}")

    # 3. Process Non-Photos (Train & Val)
    # A) HTML Chrome screenshots
    html_temps = make_html_templates()
    for idx, (html, w, h) in enumerate(html_temps):
        target_path = TRAIN_NON_PHOTO / f"train_chrome_page_{idx:03d}.png"
        render_chrome_html(html, target_path, width=w, height=h)

    # B) Charts, Slides, Certificates, Posters
    np_idx = len(html_temps)
    for s in range(60):
        # Charts
        c_img = draw_pillow_chart(seed=s)
        c_img.save(TRAIN_NON_PHOTO / f"train_chart_{np_idx:04d}.png")
        np_idx += 1

        # Slides
        s_img = draw_pillow_slide(seed=s)
        s_img.save(TRAIN_NON_PHOTO / f"train_slide_{np_idx:04d}.png")
        np_idx += 1

        # Posters / Logos
        p_img = draw_pillow_poster_logo(seed=s)
        p_img.save(TRAIN_NON_PHOTO / f"train_poster_{np_idx:04d}.png")
        np_idx += 1

    # C) Certificates from build_graphics_dataset
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_graphics_dataset import draw_certificate, draw_text_document
        for c in range(50):
            cert = draw_certificate(c)
            cert.save(TRAIN_NON_PHOTO / f"train_cert_{np_idx:04d}.png")
            np_idx += 1
            doc = draw_text_document(c)
            doc.save(TRAIN_NON_PHOTO / f"train_doc_{np_idx:04d}.png")
            np_idx += 1
    except Exception as e:
        logger.warning(f"Could not import build_graphics_dataset helpers: {e}")

    logger.info(f"Saved {np_idx} training non-photo samples to {TRAIN_NON_PHOTO}")

    # 4. Process Val Non-Photos
    vnp_idx = 0
    # Copy from existing held-out graphics_test
    graphics_test_dir = DATA_DIR / "graphics_test"
    if graphics_test_dir.exists():
        for p in graphics_test_dir.iterdir():
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                shutil.copy(p, VAL_NON_PHOTO / f"val_graphics_{p.name}")
                vnp_idx += 1

    # Add extra unseen validation non-photos
    for s in range(200, 225):
        c_img = draw_pillow_chart(seed=s)
        c_img.save(VAL_NON_PHOTO / f"val_chart_{vnp_idx:04d}.png")
        vnp_idx += 1

        s_img = draw_pillow_slide(seed=s)
        s_img.save(VAL_NON_PHOTO / f"val_slide_{vnp_idx:04d}.png")
        vnp_idx += 1

    logger.info(f"Saved {vnp_idx} validation non-photo samples to {VAL_NON_PHOTO}")
    logger.info("Gate dataset creation complete.")


if __name__ == "__main__":
    build_gate_dataset()
