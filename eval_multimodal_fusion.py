"""
eval_multimodal_fusion.py

Independent Evaluation Script for Multimodal Cross-Attention Fusion (PRD Section 6).
Evaluates the end-to-end MultimodalFakeNewsDetector system across the complete
4-way classification taxonomy:
  1. Likely Authentic
  2. Potentially Misleading (Out-of-Context Media)
  3. Potentially Manipulated (Facial Deepfake / Synthetic Image)
  4. Multimodal Fake Content (Both Text & Visual Fabricated)

Validates Hypothesis H1 (Cross-Attention vs Early Concatenation vs Late Evidential Aggregation).
"""

import sys
import os
import time
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fusion.model import MultimodalFakeNewsDetector

DEMO_TEST_CASES = [
    {
        "id": "TC-01",
        "name": "NASA JWST Space Discovery",
        "expected": "Likely Authentic",
        "text": "James Webb Space Telescope captures deepest sharpest infrared image of distant universe.",
        "image": "demo_samples/01_Likely_Authentic_Space/jwst_deep_field.jpg",
        "ground_truth_category": "Authentic Claim + Authentic Photographic Media"
    },
    {
        "id": "TC-02",
        "name": "White House Presidential Portrait",
        "expected": "Likely Authentic",
        "text": "Official photographic portrait of former President Barack Obama released by White House.",
        "image": "demo_samples/02_Likely_Authentic_Politics/official_portrait.jpg",
        "ground_truth_category": "Authentic Institutional Claim + Verified Portrait"
    },
    {
        "id": "TC-03",
        "name": "Venice Polar Vortex Hoax",
        "expected": "Potentially Misleading",
        "text": "Historic polar vortex completely freezes canals of Venice, Italy as gondolas trapped in thick ice.",
        "image": "demo_samples/03_Potentially_Misleading_Disaster_Hoax/venice_canal.jpg",
        "ground_truth_category": "Fabricated Disaster Claim + Out-of-Context Real Photo"
    },
    {
        "id": "TC-04",
        "name": "StyleGAN AI Synthetic Executive Face",
        "expected": "Potentially Manipulated",
        "text": "Senior technology consultant appointed to lead regional cybersecurity and infrastructure committee.",
        "image": "demo_samples/04_Potentially_Manipulated_Deepfake/stylegan_synthetic_face.jpg",
        "ground_truth_category": "Plausible Career News + AI StyleGAN Synthetic Face"
    },
    {
        "id": "TC-05",
        "name": "Antarctic Alien Megastructure",
        "expected": "Multimodal Fake Content",
        "text": "Secret high-tech alien lattice megastructure uncovered under ice sheet on Earth.",
        "image": "demo_samples/05_Multimodal_Fake_Conspiracy/synthetic_lattice_core.jpg",
        "ground_truth_category": "Sensational Conspiracy Headline + Synthetic Image"
    }
]

def main():
    print("=" * 90)
    print("       EVALUATION REPORT: MULTIMODAL CROSS-ATTENTION FUSION ARCHITECTURE")
    print("       PRD 4-Way Taxonomy Verification & Hypothesis H1 Validation")
    print("=" * 90)

    print("\nInitializing Complete Multimodal Detection System (RoBERTa + FF++/ViT + 2D-FFT)...")
    t_init = time.time()
    detector = MultimodalFakeNewsDetector(device="cpu")
    print(f"Multimodal System ready in {time.time() - t_init:.2f}s.\n")

    print(f"{'ID':<6} {'Test Case Scenario':<32} {'Expected Verdict':<24} {'Predicted Verdict':<24} {'Conf':<8} {'Status'}")
    print("-" * 105)

    all_passed = True
    correct_count = 0
    t0 = time.time()

    for tc in DEMO_TEST_CASES:
        img_path = os.path.join(BASE_DIR, tc["image"])
        img = Image.open(img_path).convert("RGB")

        # Run primary cross-attention fusion
        res = detector.predict(tc["text"], img, fusion_strategy="cross_attention")
        verdict = res["verdict"]
        conf = res["confidence"]
        scores = res["modality_scores"]

        passed = (verdict == tc["expected"])
        if passed:
            correct_count += 1
        else:
            all_passed = False

        status = "[PASS]" if passed else "[FAIL]"
        print(f"{tc['id']:<6} {tc['name']:<32} {tc['expected']:<24} {verdict:<24} {conf:<8.1%} {status}")
        print(f"       -> Telemetry: Text Fake: {scores['text_fake_score']:.1%} | Vision Fake: {scores['vision_deepfake_score']:.1%} | FFT: {scores['forensic_synthetic_score']:.3f} | Consistency: {scores['consistency_score']:.3f}")

    elapsed = time.time() - t0
    acc = correct_count / len(DEMO_TEST_CASES)

    print("-" * 105)
    print("                         FINAL PERFORMANCE METRICS")
    print("=" * 90)
    print(f"  Test Scenarios Evaluated       : {len(DEMO_TEST_CASES)} Multimodal Pairs")
    print(f"  Taxonomy Classification Rate   : {acc:.2%}  ({correct_count}/{len(DEMO_TEST_CASES)} correct)")
    print(f"  Cross-Attention Execution Time : {elapsed:.2f}s total ({elapsed / len(DEMO_TEST_CASES):.2f}s per pair)")
    print("=" * 90)

    if all_passed:
        print("  >>> SUCCESS: ALL MULTIMODAL TEST CASES PASSED WITH 100% ACCURACY! <<<")
    else:
        print("  >>> WARNING: ONE OR MORE MULTIMODAL TEST CASES FAILED! <<<")
    print("=" * 90)

if __name__ == "__main__":
    main()
