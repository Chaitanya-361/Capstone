"""
System Integration Test Suite
Validates all branches, tensor shapes, and fusion contracts.
"""

import sys
import os
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_branch.model import PretrainedTextBranch
from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch
from fusion.consistency import CrossModalConsistency
from fusion.cross_attention import CrossAttentionFusion
from fusion.model import MultimodalFakeNewsDetector
from video_branch.video_analyzer import VideoAnalyzer


def test_system():
    print("=" * 60)
    print("  Multimodal Fake News Detection System — Verification")
    print("=" * 60)

    # 1. Text Branch
    print("\n[1/5] Testing Text Branch (RoBERTa)...")
    text_branch = PretrainedTextBranch()
    t_res = text_branch.analyze("Scientists discover habitable exoplanet")
    assert t_res["features"].shape == (768,), f"Expected (768,), got {t_res['features'].shape}"
    assert 0.0 <= t_res["fake_prob"] <= 1.0
    print("  [OK] Text Branch PASS! Features (768,)")

    # 2. Vision Branch
    print("\n[2/5] Testing Vision Branch (EfficientNet-B0 FF++)...")
    vision_branch = VisionBranch()
    dummy_img = Image.new("RGB", (224, 224), color=(100, 120, 140))
    v_res = vision_branch.analyze(dummy_img)
    assert v_res["features"].shape == (768,), f"Expected (768,), got {v_res['features'].shape}"
    assert 0.0 <= v_res["deepfake_prob"] <= 1.0
    print("  [OK] Vision Branch PASS! Features (768,)")

    # 3. Forensic Branch
    print("\n[3/5] Testing Forensic Branch (2D-FFT)...")
    forensic_branch = ForensicBranch()
    f_res = forensic_branch.analyze(dummy_img)
    assert f_res["features"].shape == (128,), f"Expected (128,), got {f_res['features'].shape}"
    assert 0.0 <= f_res["synthetic_score"] <= 1.0
    assert f_res["spectrum_image"] is not None
    print("  [OK] Forensic Branch PASS! Features (128,)")

    # 4. Cross-Modal Consistency & Cross-Attention
    print("\n[4/5] Testing Cross-Attention Fusion & Consistency...")
    consistency_branch = CrossModalConsistency()
    c_res = consistency_branch.compute(t_res["features"], v_res["features"])
    assert 0.0 <= c_res["consistency_score"] <= 1.0

    fusion = CrossAttentionFusion()
    t_ten = torch.tensor(t_res["features"]).unsqueeze(0)
    v_ten = torch.tensor(v_res["features"]).unsqueeze(0)
    f_ten = torch.tensor(f_res["features"]).unsqueeze(0)
    c_ten = torch.tensor([[c_res["consistency_score"]]])
    f_out = fusion(t_ten, v_ten, f_ten, c_ten)
    assert f_out["logits"].shape == (1, 4), f"Expected (1, 4), got {f_out['logits'].shape}"
    print("  [OK] Cross-Attention Fusion PASS! Logits (1, 4)")

    # 5. Full End-to-End Detector
    print("\n[5/5] Testing End-to-End Detector...")
    detector = MultimodalFakeNewsDetector()
    for strategy in ["cross_attention", "early", "late"]:
        res = detector.predict("Breaking news headline", dummy_img, fusion_strategy=strategy)
        assert res["verdict"] in detector.CATEGORIES
        assert len(res["probabilities"]) == 4
        print(f"  [OK] Strategy '{strategy}': Verdict = {res['verdict']} (Confidence = {res['confidence']:.1%})")

    # 6. Video Branch (VideoAnalyzer)
    print("\n[6/6] Testing Video Branch (VideoAnalyzer)...")
    import tempfile
    import cv2
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        dummy_video_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dummy_video_path, fourcc, 10.0, (224, 224))
    for i in range(20):
        frame = np.full((224, 224, 3), 120 + i, dtype=np.uint8)
        out.write(frame)
    out.release()

    video_analyzer = VideoAnalyzer(detector=detector, num_frames=8)
    v_res = video_analyzer.analyze(dummy_video_path, text="Testing video pipeline", fusion_strategy="cross_attention")
    assert "video" in v_res
    assert "video_risk" in v_res["video"]
    assert "key_frame" in v_res
    print(f"  [OK] Video Branch PASS! Video Risk = {v_res['video']['video_risk']:.1%}")
    if os.path.exists(dummy_video_path):
        os.remove(dummy_video_path)

    print("\n" + "=" * 60)
    print("  ALL 6/6 INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_system()
