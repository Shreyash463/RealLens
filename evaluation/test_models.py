"""Script to verify that candidate Hugging Face detectors exist and load."""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Create a sample test image
test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

models_to_test = [
    "umm-maybe/AI-image-detector",
    "dima806/ai_vs_real_image_detection",
    "Ateeqq/ai-vs-human-image-detector",
    "Organika/sdxl-detector",
]

print("Testing candidate models...", flush=True)

for model_name in models_to_test:
    print(f"\n--- Testing {model_name} ---", flush=True)
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()

        inputs = processor(images=test_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

        id2label = getattr(model.config, "id2label", {})
        print(f"SUCCESS: {model_name}")
        print(f"id2label: {id2label}")
        print(f"Output probs: {probs}")
    except Exception as e:
        print(f"FAILED {model_name}: {e}")
