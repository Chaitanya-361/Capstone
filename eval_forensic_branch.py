"""
eval_forensic_branch.py

Independent Evaluation Script for the Forensic Frequency Spectrum Branch.
Method: 2D Fast Fourier Transform (2D-FFT) Azimuthal Power Spectrum Analysis.
Detects: GAN upsampling periodic artifacts, high-frequency checkerboard grid noise,
and departures from the natural 1/f^p spectral falloff law.

Dataset: 100 Frequency Power Spectra
  - 50 Natural Photographic Spectra (Decaying smoothly from low to high frequency)
  - 50 Synthetic Media Spectra (Periodic lattice noise and elevated high-frequency peaks)
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

from forensic_branch.fft_detector import ForensicBranch
from eval_vision_branch import load_evaluation_image_pool

def main():
    print("=" * 80)
    print("          EVALUATION REPORT: FORENSIC FREQUENCY SPECTRUM (2D-FFT)")
    print("          Method: 2D-FFT Azimuthal Integration + High-Frequency Energy Ratio")
    print("=" * 80)

    real_pool, fake_pool = load_evaluation_image_pool()
    print(f"Total Evaluation Samples: {len(real_pool) + len(fake_pool)}")
    print(f"  - Natural Photographic Spectra: {len(real_pool)}")
    print(f"  - Synthetic Media Spectra     : {len(fake_pool)}")

    print("\nInitializing Forensic 2D-FFT Analyzer...")
    t_init = time.time()
    forensic_model = ForensicBranch()
    print(f"Forensic engine ready in {time.time() - t_init:.2f}s.")

    items = [(p, 0) for p in real_pool] + [(p, 1) for p in fake_pool]
    y_true = [it[1] for it in items]
    y_pred = []
    hf_energies = []

    print("\nComputing 2D Fast Fourier Transforms and Azimuthal Radial Profiles...")
    t0 = time.time()
    for idx, (img_path, true_label) in enumerate(items):
        img = Image.open(img_path).convert("RGB")
        res = forensic_model.analyze(img)
        # Synthetic score threshold: calibrated at 0.2725 for frequency lattice peaks
        score = res["synthetic_score"]
        hf_energies.append(res.get("high_freq_energy", 0.0))
        y_pred.append(1 if score >= 0.2725 else 0)

        if (idx + 1) % 25 == 0:
            print(f"  Computed FFT for {idx + 1}/100 images...")

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
    print(f"  Average FFT Speed  : {elapsed / 100 * 1000:.1f} ms / spectrum ({elapsed:.2f}s total)")
    print("-" * 80)
    print("  Spectral Energy Telemetry:")
    print(f"    - Mean Natural High-Freq Energy   : {sum(hf_energies[:50])/50:.4f}")
    print(f"    - Mean Synthetic High-Freq Energy : {sum(hf_energies[50:])/50:.4f}")
    print("-" * 80)
    print("  Confusion Matrix:")
    print(f"    - True Negatives  (Natural spectra correctly identified) : {cm[0, 0]} / {len(real_pool)}")
    print(f"    - False Positives (Natural spectra flagged as synthetic) : {cm[0, 1]} / {len(real_pool)}")
    print(f"    - False Negatives (Synthetic grids missed)               : {cm[1, 0]} / {len(fake_pool)}")
    print(f"    - True Positives  (Synthetic grids correctly flagged)    : {cm[1, 1]} / {len(fake_pool)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
