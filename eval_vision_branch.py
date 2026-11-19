"""
eval_vision_branch.py

Independent Evaluation Script for the Vision Deepfake Branch.
Backbones:
  1. EfficientNet-B0 fine-tuned on FaceForensics++ (c23 compression benchmark)
  2. Vision Transformer (ViT-base-patch16-224) for Generative AI & StyleGAN detection

Dataset: 100 Benchmark Image Samples
  - 50 Ground-Truth Authentic Images (Real portraits, natural landscapes, institutional photos)
  - 50 Ground-Truth Synthetic / Deepfake Images (StyleGAN, 140k faces, facial manipulations)
"""

import sys
import os
import glob
import time
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vision_branch.model import VisionBranch

def load_evaluation_image_pool():
    real_images = []
    fake_images = []

    # Priority Demo Presets
    demo_real = [
        os.path.join(BASE_DIR, "demo_samples", "01_Likely_Authentic_Space", "jwst_deep_field.jpg"),
        os.path.join(BASE_DIR, "demo_samples", "02_Likely_Authentic_Politics", "official_portrait.jpg"),
        os.path.join(BASE_DIR, "demo_samples", "03_Potentially_Misleading_Disaster_Hoax", "venice_canal.jpg"),
    ]
    demo_fake = [
        os.path.join(BASE_DIR, "demo_samples", "04_Potentially_Manipulated_Deepfake", "stylegan_synthetic_face.jpg"),
        os.path.join(BASE_DIR, "demo_samples", "05_Multimodal_Fake_Conspiracy", "synthetic_lattice_core.jpg"),
    ]

    for p in demo_real:
        if os.path.exists(p):
            real_images.append(p)
    for p in demo_fake:
        if os.path.exists(p):
            fake_images.append(p)

    # Add from curated benchmark pool
    candidate_real = sorted(glob.glob(os.path.join(BASE_DIR, "data", "image_eval", "real", "*.jpg")))
    candidate_fake = sorted(glob.glob(os.path.join(BASE_DIR, "data", "image_eval", "fake", "*.jpg")))

    # Fallback to 140k test pool if needed
    if len(candidate_real) < 47:
        candidate_real += sorted(glob.glob(os.path.join(BASE_DIR, "data", "140k_faces", "test", "real", "*.jpg")))
    if len(candidate_fake) < 48:
        candidate_fake += sorted(glob.glob(os.path.join(BASE_DIR, "data", "140k_faces", "test", "fake", "*.jpg")))

    for p in candidate_real:
        if len(real_images) >= 50:
            break
        if p not in real_images:
            real_images.append(p)

    for p in candidate_fake:
        if len(fake_images) >= 50:
            break
        if p not in fake_images:
            fake_images.append(p)

    return real_images[:50], fake_images[:50]

def main():
    print("=" * 80)
    print("          EVALUATION REPORT: VISION DEEPFAKE BRANCH")
    print("          Backbones: EfficientNet-B0 (FF++ c23) + ViT-base-patch16-224")
    print("=" * 80)

    real_pool, fake_pool = load_evaluation_image_pool()
    print(f"Total Evaluation Samples: {len(real_pool) + len(fake_pool)}")
    print(f"  - Ground-Truth Authentic Images: {len(real_pool)}")
    print(f"  - Ground-Truth Deepfake Images : {len(fake_pool)}")

    print("\nInitializing Pretrained Vision Multi-Backbone Ensemble...")
    t_init = time.time()
    vision_model = VisionBranch(device="cpu")
    print(f"Model initialized in {time.time() - t_init:.2f}s.")

    items = [(p, 0) for p in real_pool] + [(p, 1) for p in fake_pool]
    y_true = [it[1] for it in items]
    y_pred = []
    scores = []

    print("\nRunning visual inference on 100 test images...")
    t0 = time.time()
    for idx, (img_path, true_label) in enumerate(items):
        res = vision_model.analyze(img_path)
        prob = res["deepfake_prob"]
        scores.append(prob)
        y_pred.append(1 if prob >= 0.50 else 0)

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/100 images...")

    elapsed = time.time() - t0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 80)
    print("                         FINAL PERFORMANCE METRICS")
    print("=" * 80)
    print(f"  Accuracy           : {acc:.2%}  ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/100 correct)")
    print(f"  Precision (Fake)   : {prec:.2%}")
    print(f"  Recall (Fake)      : {rec:.2%}")
    print(f"  F1-Score           : {f1:.2%}")
    print(f"  Inference Speed    : {elapsed / 100 * 1000:.1f} ms / image ({elapsed:.2f}s total)")
    print("-" * 80)
    print("  Confusion Matrix:")
    print(f"    - True Negatives  (Authentic correctly identified) : {cm[0, 0]} / {len(real_pool)}")
    print(f"    - False Positives (Authentic incorrectly flagged)  : {cm[0, 1]} / {len(real_pool)}")
    print(f"    - False Negatives (Deepfakes missed)               : {cm[1, 0]} / {len(fake_pool)}")
    print(f"    - True Positives  (Deepfakes correctly flagged)    : {cm[1, 1]} / {len(fake_pool)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
