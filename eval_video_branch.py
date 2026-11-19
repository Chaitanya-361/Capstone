"""
eval_video_branch.py

Independent Evaluation Script for the Temporal Video Deepfake Branch.
Subsystem: Multi-Frame Temporal Video Stream Analysis & Deepfake Tracking.
Backbones:
  - Frame Extraction & Sampling (5-8 frames per video stream)
  - OpenCV YuNet ONNX Deep Neural Face Tracker (face_detection_yunet_2023mar.onnx)
  - Spatial & Frequency Deepfake Risk Ensemble (FF++ EfficientNet + ViT)
  - Inter-Frame Temporal Inconsistency / Jitter Analysis (sigma_Delta)

Dataset: Benchmark Video Streams (Authentic Interviews vs Deepfake Speech Streams)
"""

import sys
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch
from video_branch.video_analyzer import VideoAnalyzer

class UnifiedVideoDetector:
    def __init__(self):
        self.vision_branch = VisionBranch(device="cpu")
        self.forensic_branch = ForensicBranch()

def main():
    print("=" * 85)
    print("         EVALUATION REPORT: TEMPORAL VIDEO DEEPFAKE ANALYSIS SUBSYSTEM")
    print("         Backbones: YuNet Face Tracker + FF++/ViT Ensemble + Temporal Jitter")
    print("=" * 85)

    test_videos = [
        # Ground-Truth Authentic Streams
        ("demo_samples/06_Video_Deepfake_Demo/authentic_interview_sample.mp4", 0, "Authentic Live Interview (Obama)"),
        ("data/video_eval/real/real_video_clip_01.mp4", 0, "Authentic Broadcast Clip 01"),
        ("data/video_eval/real/real_video_clip_04.mp4", 0, "Authentic Broadcast Clip 04"),
        # Ground-Truth Deepfake Streams
        ("demo_samples/06_Video_Deepfake_Demo/deepfake_speech_sample.mp4", 1, "Deepfake AI Manipulated Speech"),
        ("data/video_eval/fake/deepfake_sample_1.mp4", 1, "Deepfake Face-Swap Sequence 01"),
        ("data/video_eval/fake/deepfake_video_clip_02.mp4", 1, "Deepfake Face-Swap Sequence 02"),
    ]

    print("\nInitializing Video Analyzer and Neural Face Tracking Engine...")
    t_init = time.time()
    detector = UnifiedVideoDetector()
    video_analyzer = VideoAnalyzer(detector=detector, num_frames=5)
    print(f"Video Forensic Engine ready in {time.time() - t_init:.2f}s.")

    y_true = []
    y_pred = []
    results = []

    print("\nDecoding video streams and analyzing temporal frame trajectories...\n")
    print(f"{'Video File':<32} {'Ground Truth':<14} {'Risk Score':<12} {'Jitter':<10} {'Verdict Status'}")
    print("-" * 85)

    t0 = time.time()
    for rel_path, true_lbl, desc in test_videos:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            continue

        res = video_analyzer.analyze(full_path)["video"]
        risk = res["video_risk"]
        flagged = res["is_flagged"]
        jitter = res["temporal_jitter"]
        pred_lbl = 1 if flagged else 0

        y_true.append(true_lbl)
        y_pred.append(pred_lbl)

        gt_str = "Authentic" if true_lbl == 0 else "Deepfake"
        status_str = "[PASS]" if pred_lbl == true_lbl else "[FAIL]"
        pred_str = "FLAGGED DEEPFAKE" if flagged else "CLEAN / AUTHENTIC"

        print(f"{os.path.basename(rel_path):<32} {gt_str:<14} {risk:<12.1%} {jitter:<10.3f} {pred_str:<18} {status_str}")
        results.append({
            "name": desc,
            "path": rel_path,
            "true": true_lbl,
            "pred": pred_lbl,
            "risk": risk,
            "jitter": jitter,
            "frames": res["total_frames_sampled"]
        })

    elapsed = time.time() - t0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("-" * 85)
    print("                         FINAL PERFORMANCE METRICS")
    print("=" * 85)
    print(f"  Total Video Streams Evaluated : {len(y_true)} Video Sequences")
    print(f"  Classification Accuracy       : {acc:.2%}  ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)} streams correct)")
    print(f"  Precision (Deepfake)          : {prec:.2%}")
    print(f"  Recall (Deepfake)             : {rec:.2%}")
    print(f"  F1-Score                      : {f1:.2%}")
    print(f"  Average Processing Speed      : {elapsed / len(y_true):.2f} s / video stream")
    print("=" * 85)

if __name__ == "__main__":
    main()
