"""
evaluate_accuracy_benchmark.py

Comprehensive evaluation script testing:
1. Pretrained FaceForensics++ EfficientNet-B0 (Spatial CNN)
2. Pretrained Vision Transformer (ViT) Deepfake Detector
3. 2D-FFT Forensic Frequency Spectrum Detector
4. Multi-frame Video Analyzer on Temporal Video Clips
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch
from video_branch.video_analyzer import VideoAnalyzer
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(BASE_DIR, "data", "eval_small_benchmark")

print("=" * 70)
print("   MULTIMODAL DEEPFAKE & VIDEO ACCURACY BENCHMARK REPORT")
print("=" * 70)

# -------------------------------------------------------------
# 1. Image Evaluation
# -------------------------------------------------------------
print("\n[PART 1] Evaluating Pretrained Deepfake Detectors on Image Benchmark...")
real_images = sorted(glob.glob(os.path.join(BENCH_DIR, "images", "real", "*.*")))
fake_images = sorted(glob.glob(os.path.join(BENCH_DIR, "images", "fake", "*.*")))

print(f"Total Image Samples: {len(real_images)} Real, {len(fake_images)} Deepfake")

vision_ffpp = VisionBranch(device="cpu")
forensic_fft = ForensicBranch()

vit_path = os.path.join(BASE_DIR, "models", "pretrained_deepfake")
vit_processor = AutoImageProcessor.from_pretrained(vit_path)
vit_model = AutoModelForImageClassification.from_pretrained(vit_path)
vit_model.eval()

y_true = [0] * len(real_images) + [1] * len(fake_images)
all_images = real_images + fake_images

ffpp_preds = []
vit_preds = []
fft_preds = []
ensemble_preds = []

for img_path in all_images:
    img = Image.open(img_path).convert("RGB")
    
    # 1. FF++ EfficientNet-B0
    v_res = vision_ffpp.analyze(img)
    v_prob = v_res["deepfake_prob"]
    ffpp_preds.append(v_prob)
    
    # 2. Pretrained ViT
    inputs = vit_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = vit_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].numpy()
        vit_prob = float(probs[1])  # 1 is Fake
    vit_preds.append(vit_prob)
    
    # 3. 2D-FFT Forensic
    f_res = forensic_fft.analyze(img)
    fft_prob = f_res["synthetic_score"]
    fft_preds.append(fft_prob)
    
    # Ensemble (Weighted average: 0.55 FF++ + 0.45 ViT)
    ens_prob = 0.55 * v_prob + 0.45 * vit_prob
    ensemble_preds.append(ens_prob)

def print_metrics(name, scores, targets, threshold=0.5):
    preds = [1 if s >= threshold else 0 for s in scores]
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    cm = confusion_matrix(targets, preds)
    print(f"\nModel: {name} (Threshold = {threshold})")
    print(f"  Accuracy : {acc:.1%}")
    print(f"  Precision: {prec:.1%}")
    print(f"  Recall   : {rec:.1%}")
    print(f"  F1-Score : {f1:.1%}")
    print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]} | FN={cm[1,0]}, TP={cm[1,1]}")

print_metrics("Pretrained EfficientNet-B0 (FaceForensics++ c23)", ffpp_preds, y_true)
print_metrics("Pretrained Vision Transformer (ViT)", vit_preds, y_true)
print_metrics("Multimodal Dual-Branch Ensemble (FF++ + ViT)", ensemble_preds, y_true)

# -------------------------------------------------------------
# 2. Video Evaluation
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("[PART 2] Evaluating Multi-Frame Pretrained Video Deepfake Analyzer...")
real_videos = sorted(glob.glob(os.path.join(BENCH_DIR, "videos", "real", "*.*")))
fake_videos = sorted(glob.glob(os.path.join(BENCH_DIR, "videos", "fake", "*.*")))
print(f"Total Video Samples: {len(real_videos)} Real, {len(fake_videos)} Deepfake\n")

class VideoDetectorWrapper:
    def __init__(self, v, f):
        self.vision_branch = v
        self.forensic_branch = f

wrapper = VideoDetectorWrapper(vision_ffpp, forensic_fft)
video_analyzer = VideoAnalyzer(detector=wrapper, num_frames=8)

vid_results = []
for v_path in real_videos:
    res = video_analyzer.analyze(v_path)
    v_meta = res["video"]
    vid_results.append({
        "file": os.path.basename(v_path),
        "true_label": 0,
        "risk": v_meta["video_risk"],
        "jitter": v_meta["temporal_jitter"],
        "flagged_ratio": v_meta["flagged_frame_ratio"],
        "is_flagged": v_meta["is_flagged"]
    })

for v_path in fake_videos:
    res = video_analyzer.analyze(v_path)
    v_meta = res["video"]
    vid_results.append({
        "file": os.path.basename(v_path),
        "true_label": 1,
        "risk": v_meta["video_risk"],
        "jitter": v_meta["temporal_jitter"],
        "flagged_ratio": v_meta["flagged_frame_ratio"],
        "is_flagged": v_meta["is_flagged"]
    })

print(f"{'Video File':<25} {'True':<7} {'Risk':<9} {'Jitter':<9} {'Flagged %':<11} {'Prediction':<10}")
print("-" * 75)
correct_vids = 0
for r in vid_results:
    true_str = "Real" if r["true_label"] == 0 else "Fake"
    pred_str = "Fake" if r["is_flagged"] else "Real"
    is_corr = (r["true_label"] == 1 and r["is_flagged"]) or (r["true_label"] == 0 and not r["is_flagged"])
    if is_corr:
        correct_vids += 1
    mark = "[OK]" if is_corr else "[FAIL]"
    print(f"{r['file']:<25} {true_str:<7} {r['risk']:<9.1%} {r['jitter']:<9.4f} {r['flagged_ratio']:<11.0%} {pred_str:<10} {mark}")

print(f"\nVideo Deepfake Detection Accuracy: {correct_vids}/{len(vid_results)} = {correct_vids/len(vid_results):.1%}")
print("=" * 70)
