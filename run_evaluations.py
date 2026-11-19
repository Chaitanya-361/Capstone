"""
run_evaluations.py

Unified Master Evaluation Suite for Multimodal Fake News & Deepfake Detection.
Academic Capstone Project — SCOPE, VIT-AP University (2026).

Usage:
  python run_evaluations.py --branch text        # Evaluate Text Branch (100 samples)
  python run_evaluations.py --branch vision      # Evaluate Vision Deepfake Branch (100 samples)
  python run_evaluations.py --branch forensic    # Evaluate Forensic 2D-FFT Branch (100 samples)
  python run_evaluations.py --branch video       # Evaluate Video Deepfake Branch
  python run_evaluations.py --branch fusion      # Evaluate Multimodal Cross-Attention Fusion
  python run_evaluations.py --branch all         # Run all branch evaluations sequentially
"""

import sys
import os
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "text": "eval_text_branch.py",
    "vision": "eval_vision_branch.py",
    "forensic": "eval_forensic_branch.py",
    "video": "eval_video_branch.py",
    "fusion": "eval_multimodal_fusion.py",
}

def run_script(branch_name):
    script_file = SCRIPTS.get(branch_name)
    if not script_file:
        print(f"Unknown branch: {branch_name}")
        return False

    script_path = os.path.join(BASE_DIR, script_file)
    print("\n" + "#" * 80)
    print(f"  EXECUTING: {script_file}  (--branch {branch_name})")
    print("#" * 80 + "\n")

    cmd = [sys.executable, script_path]
    res = subprocess.run(cmd)
    return res.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Suite for Multimodal Fake News Detection")
    parser.add_argument(
        "--branch",
        type=str,
        default="menu",
        choices=["text", "vision", "forensic", "video", "fusion", "all", "menu"],
        help="Specify which branch evaluation to run."
    )
    args = parser.parse_args()

    if args.branch == "menu":
        print("=" * 80)
        print("     MULTIMODAL DEEPFAKE & FAKE NEWS DETECTION — BENCHMARK SUITE")
        print("=" * 80)
        print("Select a branch evaluation to run:")
        print("  1. Text Branch (RoBERTa-base, 100 Claims)")
        print("  2. Vision Deepfake Branch (FF++ EfficientNet + ViT, 100 Images)")
        print("  3. Forensic Frequency Spectrum (2D-FFT, 100 Spectra)")
        print("  4. Video Deepfake Analyzer (YuNet Face Tracker + Temporal Streams)")
        print("  5. Multimodal Cross-Attention Fusion (PRD 4-Way Taxonomy)")
        print("  6. Run All Branches Sequentially")
        print("  0. Exit")
        print("-" * 80)

        choice = input("Enter choice [1-6]: ").strip()
        choice_map = {
            "1": "text",
            "2": "vision",
            "3": "forensic",
            "4": "video",
            "5": "fusion",
            "6": "all",
        }
        branch = choice_map.get(choice)
        if not branch:
            print("Exiting.")
            return
    else:
        branch = args.branch

    if branch == "all":
        for b in ["text", "vision", "forensic", "video", "fusion"]:
            run_script(b)
    else:
        run_script(branch)

if __name__ == "__main__":
    main()
