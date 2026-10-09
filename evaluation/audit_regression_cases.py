"""Audit script to inspect and reproduce model predictions on the 2 user regression cases."""
from pathlib import Path
from PIL import Image
import os
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reallens.detectors.ml_ensemble import MLEnsemble
from reallens.detectors.image_type import classify_image_type
from reallens.detectors.fusion import fuse_signals
from reallens.detectors.metadata import analyze_metadata
from reallens.detectors.c2pa_check import analyze_c2pa
from reallens.detectors.ela import analyze_ela
from reallens.detectors.frequency import analyze_frequency

ai_path = Path("data/my_images/ai/oiuhygfghjk.jpeg")
real_path = Path("data/my_images/real/WhatsApp Image 2026-10-09 at 12.45.33 PM.jpeg")

ens = MLEnsemble()
ens.load_all()

for label, p in [("Image 1 (Ganesha AI)", ai_path), ("Image 2 (Man near tree Real)", real_path)]:
    print(f"\n=======================================================")
    print(f"Audit: {label}")
    print(f"Path: {p}")
    img = Image.open(p)
    raw = p.read_bytes()
    print(f"Resolution: {img.size[0]}x{img.size[1]}, Size: {len(raw)} bytes")
    
    meta = analyze_metadata(raw)
    c2pa = analyze_c2pa(raw)
    ela = analyze_ela(img)
    freq = analyze_frequency(img)
    gate = classify_image_type(img)
    ml = ens.analyze(img)
    
    print(f"Gate image_type: {gate['image_type']} (photo_prob: {gate['photo_probability']})")
    print(f"ML scores:")
    for m, sc in ml['per_model_scores'].items():
        print(f"  - {m}: {sc}")
    print(f"ML spread: {ml['spread']}")
    
    verdict = fuse_signals(meta, c2pa, ela, freq, ml, image_type_res=gate)
    print(f"Final Verdict: {verdict['verdict']} (headline: {verdict['headline']}, is_inconclusive: {verdict['is_inconclusive']})")
    print(f"Reason: {verdict['why_this_verdict']}")
    print(f"Summary: {verdict['summary_reason']}")
