"""
verify_demo_presets.py
Tests MultimodalFakeNewsDetector on all standard demo presets to verify high accuracy and confidence.
"""

import os
from PIL import Image
from fusion.model import MultimodalFakeNewsDetector

detector = MultimodalFakeNewsDetector(device="cpu")

test_cases = [
    {
        "name": "1. NASA JWST (Likely Authentic)",
        "expected": "Likely Authentic",
        "text": "James Webb Space Telescope captures deepest sharpest infrared image of distant universe.",
        "image": "demo_samples/01_Likely_Authentic_Space/jwst_deep_field.jpg"
    },
    {
        "name": "2. Official Portrait (Likely Authentic)",
        "expected": "Likely Authentic",
        "text": "Official photographic portrait of former President Barack Obama released by White House.",
        "image": "demo_samples/02_Likely_Authentic_Politics/official_portrait.jpg"
    },
    {
        "name": "3. Venice Canal Hoax (Potentially Misleading)",
        "expected": "Potentially Misleading",
        "text": "Historic polar vortex completely freezes canals of Venice, Italy as gondolas trapped in thick ice.",
        "image": "demo_samples/03_Potentially_Misleading_Disaster_Hoax/venice_canal.jpg"
    },
    {
        "name": "4. StyleGAN Synthetic Face (Potentially Manipulated)",
        "expected": "Potentially Manipulated",
        "text": "Senior technology consultant appointed to lead regional cybersecurity and infrastructure committee.",
        "image": "demo_samples/04_Potentially_Manipulated_Deepfake/stylegan_synthetic_face.jpg"
    },
    {
        "name": "5. Alien Megastructure (Multimodal Fake Content)",
        "expected": "Multimodal Fake Content",
        "text": "Secret high-tech alien lattice megastructure uncovered under ice sheet on Earth.",
        "image": "demo_samples/05_Multimodal_Fake_Conspiracy/synthetic_lattice_core.jpg"
    }
]

print("\n" + "=" * 80)
print("     VERIFYING DEMO PRESETS ON MULTIMODAL DETECTOR")
print("=" * 80)

all_passed = True
for tc in test_cases:
    img = Image.open(tc["image"]).convert("RGB")
    res = detector.predict(tc["text"], img, fusion_strategy="cross_attention")
    verdict = res["verdict"]
    conf = res["confidence"]
    passed = (verdict == tc["expected"])
    if not passed:
        all_passed = False
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {tc['name']:<42} -> Got: {verdict:<24} ({conf:.1%}) | Expected: {tc['expected']}")
    scores = res["modality_scores"]
    print(f"      Text Fake: {scores['text_fake_score']:.1%} | Vision Fake: {scores['vision_deepfake_score']:.1%} | FFT Score: {scores['forensic_synthetic_score']:.3f} | Consistency: {scores['consistency_score']:.3f}")

print("=" * 80)
if all_passed:
    print("ALL CORE DEMO SAMPLES PASSED WITH 100% ACCURACY!")
else:
    print("SOME DEMO SAMPLES FAILED!")
print("=" * 80)
