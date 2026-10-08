"""Generator for Authentic Non-AI Graphics and Screenshots Test Category.

Generates 30 diverse, real non-AI digital graphic images using Pillow and Matplotlib:
1. Academic / Corporate Certificates (borders, seals, calligraphy text, signatures)
2. Software & Web Application Screenshots (browser chrome, sidebars, data cards)
3. Business Presentation Slides (corporate layout, bullet points, geometric diagrams)
4. Formal Text Documents / Invoices (headers, tabular billing, signatures)
5. Statistical Data Charts (bar charts, line graphs, legends)
6. Flat Logos & Vector-style Posters

Also includes any user graphics placed in data/my_images/real_graphics/.
"""
from __future__ import annotations

import io
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent

def draw_certificate(idx: int) -> Image.Image:
    """Generates an authentic certificate of achievement / diploma graphic."""
    w, h = 1000, 700
    img = Image.new("RGB", (w, h), color=(253, 251, 243)) # warm ivory paper
    draw = ImageDraw.Draw(img)

    # Ornate double border
    draw.rectangle([25, 25, w - 25, h - 25], outline=(180, 140, 50), width=4)
    draw.rectangle([35, 35, w - 35, h - 35], outline=(30, 41, 59), width=2)
    draw.rectangle([42, 42, w - 42, h - 42], outline=(180, 140, 50), width=1)

    # Corner decorations
    for cx, cy in [(42, 42), (w-42, 42), (42, h-42), (w-42, h-42)]:
        draw.rectangle([cx-10, cy-10, cx+10, cy+10], outline=(180, 140, 50), fill=(245, 230, 180))

    # Certificate title & text
    draw.text((w // 2 - 180, 100), "CERTIFICATE OF ACHIEVEMENT", fill=(30, 41, 59))
    draw.text((w // 2 - 120, 140), "THIS IS PROUDLY PRESENTED TO", fill=(100, 116, 139))
    draw.text((w // 2 - 140, 200), f"RECIPIENT NAME {idx + 100}", fill=(15, 23, 42))
    draw.line([w // 2 - 200, 240, w // 2 + 200, 240], fill=(180, 140, 50), width=2)

    draw.text((w // 2 - 240, 280), "For outstanding performance and dedication in Advanced Computer Vision", fill=(71, 85, 105))
    draw.text((w // 2 - 160, 310), "Completed with distinction on October 2026", fill=(71, 85, 105))

    # Gold Seal
    seal_x, seal_y = w // 2, 460
    draw.ellipse([seal_x - 50, seal_y - 50, seal_x + 50, seal_y + 50], fill=(217, 160, 40), outline=(160, 110, 20), width=3)
    draw.ellipse([seal_x - 40, seal_y - 40, seal_x + 40, seal_y + 40], outline=(255, 235, 150), width=2)
    draw.text((seal_x - 22, seal_y - 10), "OFFICIAL", fill=(255, 255, 255))

    # Signatures
    draw.line([150, 580, 350, 580], fill=(51, 65, 85), width=2)
    draw.text((200, 595), "DIRECTOR SIGNATURE", fill=(100, 116, 139))
    draw.line([w - 350, 580, w - 150, 580], fill=(51, 65, 85), width=2)
    draw.text((w - 290, 595), "COMMITTEE CHAIR", fill=(100, 116, 139))

    return img

def draw_web_app_screenshot(idx: int) -> Image.Image:
    """Generates a desktop web application dashboard screenshot."""
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), color=(15, 23, 42)) # Dark slate theme
    draw = ImageDraw.Draw(img)

    # Window title bar (Browser chrome)
    draw.rectangle([0, 0, w, 40], fill=(30, 41, 59))
    # Window buttons
    draw.ellipse([15, 14, 27, 26], fill=(239, 68, 68)) # red
    draw.ellipse([35, 14, 47, 26], fill=(245, 158, 11)) # yellow
    draw.ellipse([55, 14, 67, 26], fill=(16, 185, 129)) # green
    # URL bar
    draw.rectangle([140, 8, w - 140, 32], fill=(15, 23, 42), outline=(51, 65, 85), width=1)
    draw.text((160, 13), f"https://dashboard.cloudsystem.internal/analytics/node-{idx}", fill=(148, 163, 184))

    # Sidebar
    draw.rectangle([0, 40, 220, h], fill=(24, 34, 53), outline=(51, 65, 85), width=1)
    draw.text((20, 60), "RealLens Cloud", fill=(248, 250, 252))
    for i, item in enumerate(["Overview", "Live Feeds", "Forensic Logs", "Security Alerts", "Settings"]):
        y_pos = 110 + i * 40
        if i == 0:
            draw.rectangle([10, y_pos - 6, 210, y_pos + 26], fill=(59, 130, 246), outline=None)
            draw.text((30, y_pos), item, fill=(255, 255, 255))
        else:
            draw.text((30, y_pos), item, fill=(148, 163, 184))

    # Main Dashboard Cards
    for c_idx, title in enumerate(["API Throughput", "Active Nodes", "Error Rate", "Memory Load"]):
        cx = 250 + (c_idx % 2) * 490
        cy = 70 + (c_idx // 2) * 160
        draw.rectangle([cx, cy, cx + 460, cy + 130], fill=(24, 34, 53), outline=(51, 65, 85), width=1)
        draw.text((cx + 20, cy + 20), title, fill=(148, 163, 184))
        draw.text((cx + 20, cy + 50), f"{random.randint(80, 99)}.{random.randint(10, 99)}%", fill=(248, 250, 252))
        draw.rectangle([cx + 20, cy + 95, cx + 430, cy + 105], fill=(30, 41, 59))
        draw.rectangle([cx + 20, cy + 95, cx + 20 + random.randint(150, 380), cy + 105], fill=(16, 185, 129))

    # Data Table
    table_y = 420
    draw.rectangle([250, table_y, w - 40, h - 30], fill=(24, 34, 53), outline=(51, 65, 85), width=1)
    draw.text((270, table_y + 15), "Recent Security Inspection Log", fill=(248, 250, 252))
    draw.line([250, table_y + 45, w - 40, table_y + 45], fill=(51, 65, 85), width=1)
    for r_idx in range(5):
        ry = table_y + 60 + r_idx * 38
        draw.text((270, ry), f"REQ-{idx:03d}-{r_idx:02d}", fill=(148, 163, 184))
        draw.text((450, ry), "GET /api/v1/telemetry", fill=(203, 213, 225))
        draw.text((750, ry), "200 OK", fill=(52, 211, 153))
        draw.text((950, ry), f"{random.randint(12, 45)} ms", fill=(148, 163, 184))

    return img

def draw_text_document(idx: int) -> Image.Image:
    """Generates an authentic business document / invoice."""
    w, h = 850, 1100 # Standard US Letter aspect ratio
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Document Header
    draw.rectangle([50, 50, 120, 120], fill=(37, 99, 235))
    draw.text((65, 75), "RL", fill=(255, 255, 255))
    draw.text((140, 65), "REAL LENS TECHNOLOGIES CORP.", fill=(15, 23, 42))
    draw.text((140, 90), "Enterprise Security Division • London & New York", fill=(100, 116, 139))

    # Invoice details
    draw.text((w - 250, 65), f"INVOICE #INV-{idx+2026}-89", fill=(15, 23, 42))
    draw.text((w - 250, 90), "Date: October 08, 2026", fill=(100, 116, 139))
    draw.line([50, 150, w - 50, 150], fill=(226, 232, 240), width=2)

    # Billing party
    draw.text((50, 180), "Billed To:", fill=(100, 116, 139))
    draw.text((50, 205), f"Department of Digital Forensic Science #{idx}", fill=(15, 23, 42))
    draw.text((50, 225), "742 Evergreen Terrace, Sector 4", fill=(71, 85, 105))

    # Table Header
    draw.rectangle([50, 290, w - 50, 330], fill=(241, 245, 249))
    draw.text((70, 305), "Description", fill=(71, 85, 105))
    draw.text((450, 305), "Qty", fill=(71, 85, 105))
    draw.text((550, 305), "Unit Price", fill=(71, 85, 105))
    draw.text((w - 150, 305), "Amount", fill=(71, 85, 105))

    # Items
    items = [
        ("AI Provenance Verification Appliance (v2.5)", 1, "$4,500.00"),
        ("Cryptographic C2PA Hardware Module", 2, "$1,200.00"),
        ("Enterprise Support & Maintenance Contract", 12, "$350.00"),
        ("Local On-Premise GPU Inference Server Setup", 1, "$1,850.00"),
    ]
    total_val = 0
    for r_i, (desc, qty, pr) in enumerate(items):
        ry = 350 + r_i * 45
        draw.text((70, ry), desc, fill=(30, 41, 59))
        draw.text((460, ry), str(qty), fill=(30, 41, 59))
        draw.text((550, ry), pr, fill=(30, 41, 59))
        draw.text((w - 150, ry), pr, fill=(30, 41, 59))
        draw.line([50, ry + 35, w - 50, ry + 35], fill=(241, 245, 249), width=1)

    # Total block
    draw.rectangle([w - 300, 560, w - 50, 640], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
    draw.text((w - 280, 575), "Subtotal:", fill=(100, 116, 139))
    draw.text((w - 150, 575), "$11,750.00", fill=(15, 23, 42))
    draw.text((w - 280, 605), "Total Due:", fill=(15, 23, 42))
    draw.text((w - 150, 605), "$11,750.00", fill=(37, 99, 235))

    # Terms & Signature
    draw.text((50, 750), "Payment Terms: Net 30 days. Wire transfer to authorized corporate bank.", fill=(100, 116, 139))
    draw.line([50, 890, 300, 890], fill=(15, 23, 42), width=2)
    draw.text((50, 905), "Authorized Finance Signature", fill=(100, 116, 139))

    return img

def draw_presentation_slide(idx: int) -> Image.Image:
    """Generates a presentation slide graphic."""
    w, h = 960, 540 # 16:9 widescreen
    img = Image.new("RGB", (w, h), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, w, 80], fill=(30, 41, 59))
    draw.text((50, 30), f"MODULE {idx + 1}: NEURAL ARCHITECTURES FOR IMAGE REASONING", fill=(255, 255, 255))

    # Slide Content Cards
    draw.text((50, 110), "Key Operational Trade-offs", fill=(15, 23, 42))

    points = [
        "1. Latent Diffusion Models introduce periodic Fourier checkerboard artifacts.",
        "2. Error Level Analysis requires JPEG compression history to establish equilibrium.",
        "3. Flat digital graphics, UI screenshots, and certificates trigger false positive vision alerts.",
        "4. Deterministic image gating provides safety against out-of-domain classification.",
    ]
    for p_i, pt in enumerate(points):
        py = 170 + p_i * 65
        draw.rectangle([50, py - 10, w - 50, py + 45], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        draw.rectangle([50, py - 10, 56, py + 45], fill=(59, 130, 246))
        draw.text((70, py + 8), pt, fill=(51, 65, 85))

    # Footer
    draw.text((50, 500), "Confidential • Internal Engineering Briefing 2026", fill=(148, 163, 184))
    draw.text((w - 100, 500), f"Slide {idx + 1}", fill=(148, 163, 184))
    return img

def draw_statistical_chart(idx: int) -> Image.Image:
    """Generates a clean statistical chart graphic with bars and gridlines."""
    w, h = 800, 600
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((w // 2 - 160, 40), f"BENCHMARK COMPARISON #{idx:02d} (ROC-AUC)", fill=(15, 23, 42))

    # Chart grid
    gx0, gy0 = 100, 120
    gx1, gy1 = 700, 500
    draw.line([gx0, gy1, gx1, gy1], fill=(100, 116, 139), width=2)
    draw.line([gx0, gy0, gx0, gy1], fill=(100, 116, 139), width=2)

    for g_y in range(gy0, gy1, 76):
        draw.line([gx0, g_y, gx1, g_y], fill=(241, 245, 249), width=1)
        val = 100 - int((g_y - gy0) / (gy1 - gy0) * 100)
        draw.text((gx0 - 45, g_y - 6), f"{val}%", fill=(148, 163, 184))

    # Bar groups
    labels = ["CIFAKE", "GenImage", "Midjourney", "DALL-E 3", "HardSet"]
    colors = [(59, 130, 246), (16, 185, 129), (245, 158, 11), (239, 68, 68), (139, 92, 246)]
    bar_w = 70
    for b_i, (lbl, col) in enumerate(zip(labels, colors)):
        bx = gx0 + 40 + b_i * 115
        h_val = random.randint(180, 340)
        by = gy1 - h_val
        draw.rectangle([bx, by, bx + bar_w, gy1 - 1], fill=col)
        draw.text((bx + 10, by - 22), f"{h_val // 4}%", fill=(30, 41, 59))
        draw.text((bx + 5, gy1 + 15), lbl, fill=(71, 85, 105))

    return img

def build_graphics_dataset():
    random.seed(42)
    data_dir = BASE_DIR / "data"
    out_dir = data_dir / "graphics_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in out_dir.glob("*.*"):
        f.unlink()

    print("=" * 65)
    print("  BUILDING AUTHENTIC NON-AI GRAPHICS & SCREENSHOTS TEST SET")
    print("=" * 65)

    # 1. Generate synthetic real graphics
    generators = [
        ("cert", draw_certificate, 6),
        ("screen", draw_web_app_screenshot, 6),
        ("doc", draw_text_document, 6),
        ("slide", draw_presentation_slide, 6),
        ("chart", draw_statistical_chart, 6),
    ]

    total_created = 0
    for prefix, fn, count in generators:
        for i in range(count):
            img = fn(i)
            # Save as PNG or high-quality JPEG
            ext = ".png" if i % 2 == 0 else ".jpg"
            out_path = out_dir / f"real_graphic_{prefix}_{i:02d}{ext}"
            if ext == ".png":
                img.save(out_path, format="PNG")
            else:
                img.save(out_path, format="JPEG", quality=92)
            total_created += 1

    print(f"Generated {total_created} synthetic non-AI graphics/screenshots in {out_dir}")

    # 2. Check for user-provided real graphics in data/my_images/real_graphics/
    user_graphics_dir = data_dir / "my_images" / "real_graphics"
    if user_graphics_dir.exists():
        user_files = list(user_graphics_dir.glob("*.*"))
        if user_files:
            print(f"Found {len(user_files)} user-provided graphics in {user_graphics_dir}")
            for u_f in user_files:
                shutil.copy2(u_f, out_dir / f"user_{u_f.name}")
                total_created += 1

    print(f"[SUCCESS] Total Graphics Test Set: {len(list(out_dir.glob('*.*')))} images.")

if __name__ == "__main__":
    build_graphics_dataset()
