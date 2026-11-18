"""
evaluate_media.py

Benchmark evaluation script for Vision (EfficientNet-B0), Forensic (2D-FFT),
and Video branches.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and generates Confusion Matrix plots.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch
from video_branch.video_analyzer import VideoAnalyzer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_threshold_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (np.array(y_scores) >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "threshold": threshold}


def find_best_threshold(y_true, y_scores):
    best_f1 = -1.0
    best_th = 0.5
    for th in np.linspace(0.05, 0.95, 91):
        m = evaluate_threshold_metrics(y_true, y_scores, threshold=th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
    return float(best_th)


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real (0)', 'Fake (1)'])
    plt.yticks(tick_marks, ['Real (0)', 'Fake (1)'])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def run_image_eval(vision, forensic, limit=1000, out_dir="results"):
    print(f"\n--- Running Image Evaluation (limit={limit} per class) ---")
    real_files = (glob.glob(os.path.join(BASE_DIR, "data/image_eval/real/*.jpg")) + 
                  glob.glob(os.path.join(BASE_DIR, "data/image_eval/real/*.png")) + 
                  glob.glob(os.path.join(BASE_DIR, "data/image_eval/real/*.jpeg")))[:limit]
    fake_files = (glob.glob(os.path.join(BASE_DIR, "data/image_eval/fake/*.jpg")) + 
                  glob.glob(os.path.join(BASE_DIR, "data/image_eval/fake/*.png")) + 
                  glob.glob(os.path.join(BASE_DIR, "data/image_eval/fake/*.jpeg")))[:limit]

    if not real_files and not fake_files:
        print("Warning: No image files found in data/image_eval/real or data/image_eval/fake.")
        return

    items = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    records = []

    for fpath, label in items:
        try:
            v_res = vision.analyze(fpath)
            v_score = float(v_res["deepfake_prob"])

            f_res = forensic.analyze(fpath)
            f_score = float(f_res["synthetic_score"])

            comb_score = max(v_score, f_score)
            records.append({
                "path": fpath,
                "label": label,
                "vision_score": v_score,
                "forensic_score": f_score,
                "combined_score": comb_score
            })
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    df = pd.DataFrame(records)
    out_dir_path = os.path.join(BASE_DIR, out_dir)
    os.makedirs(out_dir_path, exist_ok=True)
    df.to_csv(os.path.join(out_dir_path, "image_scores.csv"), index=False)
    report_and_save_metrics(df, "image", out_dir_path)


def run_video_eval(vision, forensic, limit=50, frames=8, out_dir="results"):
    print(f"\n--- Running Video Evaluation (limit={limit} per class, frames={frames}) ---")
    valid_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    real_dir = os.path.join(BASE_DIR, "data/video_eval/real")
    fake_dir = os.path.join(BASE_DIR, "data/video_eval/fake")
    
    real_videos = [
        os.path.join(real_dir, f) for f in os.listdir(real_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ][:limit] if os.path.exists(real_dir) else []
    
    fake_videos = [
        os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ][:limit] if os.path.exists(fake_dir) else []

    if not real_videos and not fake_videos:
        print("Warning: No video files found in data/video_eval/real or data/video_eval/fake.")
        return

    class LightweightDetector:
        def __init__(self, v, f):
            self.vision_branch = v
            self.forensic_branch = f

    mock_det = LightweightDetector(vision, forensic)
    analyzer = VideoAnalyzer(detector=mock_det, num_frames=frames)

    items = [(f, 0) for f in real_videos] + [(f, 1) for f in fake_videos]
    records = []

    for vpath, label in items:
        try:
            res = analyzer.analyze(vpath)
            v_meta = res["video"]
            records.append({
                "path": vpath,
                "label": label,
                "video_risk": v_meta["video_risk"],
                "flagged_ratio": v_meta["flagged_frame_ratio"],
                "temporal_jitter": v_meta["temporal_jitter"],
                "combined_score": v_meta["video_risk"]
            })
        except Exception as e:
            print(f"Error processing video {vpath}: {e}")

    df = pd.DataFrame(records)
    out_dir_path = os.path.join(BASE_DIR, out_dir)
    os.makedirs(out_dir_path, exist_ok=True)
    df.to_csv(os.path.join(out_dir_path, "video_scores.csv"), index=False)
    report_and_save_metrics(df, "video", out_dir_path)


def report_and_save_metrics(df, modality, out_dir):
    y_true = df["label"].values
    detectors = ["vision_score", "forensic_score", "combined_score"] if modality == "image" else ["combined_score"]

    print(f"\n================ Metrics Summary ({modality.upper()}) ================")
    for det in detectors:
        y_scores = df[det].values
        best_th = find_best_threshold(y_true, y_scores)
        m = evaluate_threshold_metrics(y_true, y_scores, threshold=best_th)
        cm = confusion_matrix(y_true, (np.array(y_scores) >= best_th).astype(int), labels=[0, 1])

        det_name = det.replace("_score", "").capitalize()
        print(f"[{det_name}] AUC: {m['auc']:.3f} | F1: {m['f1']:.3f} (th={best_th:.2f}) | Acc: {m['acc']:.3f} | Rec: {m['rec']:.3f} | Prec: {m['prec']:.3f}")

        plot_confusion_matrix(
            cm,
            f"{modality.capitalize()} {det_name} Confusion Matrix",
            os.path.join(out_dir, f"{modality}_{det}_confusion_matrix.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image & Video Branches.")
    parser.add_argument("--mode", choices=["image", "video", "both"], default="both")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vision_b = VisionBranch(device=device)
    forensic_b = ForensicBranch()

    if args.mode in ["image", "both"]:
        run_image_eval(vision_b, forensic_b, limit=args.limit, out_dir=args.out_dir)
    if args.mode in ["video", "both"]:
        vid_limit = min(args.limit, 100) if args.limit == 1000 else args.limit
        run_video_eval(vision_b, forensic_b, limit=vid_limit, frames=args.frames, out_dir=args.out_dir)
