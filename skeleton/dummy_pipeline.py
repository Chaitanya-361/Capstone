"""
Dummy End-to-End Skeleton Pipeline
Validates the entire data flow with random inputs/outputs before real training.

This catches:
    - Import errors / path issues
    - Tensor shape mismatches between branches
    - Interface incompatibilities before you waste hours training

Run: python skeleton/dummy_pipeline.py
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DummyTextBranch(nn.Module):
    """Simulates text branch: text → 768-dim feature + 3-class logits."""
    def __init__(self):
        super().__init__()
        self.feature_dim = 768
        self.fc = nn.Linear(self.feature_dim, 3)

    def forward(self, text_features=None):
        if text_features is None:
            text_features = torch.randn(1, self.feature_dim)
        logits = self.fc(text_features)
        return {"logits": logits, "features": text_features}


class DummyVisionBranch(nn.Module):
    """Simulates vision branch: image → 768-dim feature + 3-class logits."""
    def __init__(self):
        super().__init__()
        self.feature_dim = 768
        self.fc = nn.Linear(self.feature_dim, 3)

    def forward(self, image_features=None):
        if image_features is None:
            image_features = torch.randn(1, self.feature_dim)
        logits = self.fc(image_features)
        return {"logits": logits, "features": image_features}


class DummyForensicBranch(nn.Module):
    """Simulates forensic branch: FFT image → 128-dim feature + deepfake score."""
    def __init__(self):
        super().__init__()
        self.feature_dim = 128
        self.fc = nn.Linear(self.feature_dim, 1)  # Binary: real/deepfake

    def forward(self, forensic_features=None):
        if forensic_features is None:
            forensic_features = torch.randn(1, self.feature_dim)
        score = torch.sigmoid(self.fc(forensic_features))
        return {"deepfake_score": score, "features": forensic_features}


class DummyCLIPConsistency(nn.Module):
    """Simulates CLIP text-image consistency scoring."""
    def __init__(self):
        super().__init__()
        self.clip_dim = 512

    def forward(self, text_features=None, image_features=None):
        if text_features is None:
            text_features = torch.randn(1, self.clip_dim)
        if image_features is None:
            image_features = torch.randn(1, self.clip_dim)
        # Cosine similarity
        similarity = nn.functional.cosine_similarity(text_features, image_features, dim=-1)
        return {"consistency_score": similarity}


class DummyFusionLayer(nn.Module):
    """
    Simulates fusion: concatenate all features → classify.
    Feature dimensions: text(768) + vision(768) + forensic(128) + consistency(1) = 1665
    """
    def __init__(self):
        super().__init__()
        self.total_dim = 768 + 768 + 128 + 1  # 1665
        self.classifier = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),  # 4 final categories per PRD
        )

    def forward(self, text_feat, vision_feat, forensic_feat, consistency_score):
        fused = torch.cat([
            text_feat,
            vision_feat,
            forensic_feat,
            consistency_score.unsqueeze(-1),
        ], dim=-1)
        logits = self.classifier(fused)
        return {"logits": logits, "fused_features": fused}


def run_skeleton():
    """Run the full dummy pipeline and validate all shapes."""
    print("=" * 60)
    print("  Dummy Skeleton Pipeline — Interface Validation")
    print("=" * 60)

    batch_size = 4
    checks_passed = 0
    total_checks = 0

    def check(name, condition, details=""):
        nonlocal checks_passed, total_checks
        total_checks += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        print(f"  {status}: {name}" + (f" ({details})" if details else ""))
        if condition:
            checks_passed += 1
        return condition

    # ── 1. Text Branch ───────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("1. Text Branch")
    text_branch = DummyTextBranch()
    text_input = torch.randn(batch_size, 768)
    text_out = text_branch(text_input)

    check("Text logits shape", text_out["logits"].shape == (batch_size, 3),
          f"got {text_out['logits'].shape}")
    check("Text features shape", text_out["features"].shape == (batch_size, 768),
          f"got {text_out['features'].shape}")

    # ── 2. Vision Branch ─────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("2. Vision Branch")
    vision_branch = DummyVisionBranch()
    vision_input = torch.randn(batch_size, 768)
    vision_out = vision_branch(vision_input)

    check("Vision logits shape", vision_out["logits"].shape == (batch_size, 3),
          f"got {vision_out['logits'].shape}")
    check("Vision features shape", vision_out["features"].shape == (batch_size, 768),
          f"got {vision_out['features'].shape}")

    # ── 3. Forensic Branch ───────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("3. Forensic Branch")
    forensic_branch = DummyForensicBranch()
    forensic_input = torch.randn(batch_size, 128)
    forensic_out = forensic_branch(forensic_input)

    check("Forensic score shape", forensic_out["deepfake_score"].shape == (batch_size, 1),
          f"got {forensic_out['deepfake_score'].shape}")
    check("Forensic features shape", forensic_out["features"].shape == (batch_size, 128),
          f"got {forensic_out['features'].shape}")
    check("Forensic score range", 
          (forensic_out["deepfake_score"] >= 0).all() and (forensic_out["deepfake_score"] <= 1).all(),
          f"range [{forensic_out['deepfake_score'].min():.3f}, {forensic_out['deepfake_score'].max():.3f}]")

    # ── 4. CLIP Consistency ──────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("4. CLIP Consistency Scoring")
    clip_module = DummyCLIPConsistency()
    clip_text = torch.randn(batch_size, 512)
    clip_image = torch.randn(batch_size, 512)
    clip_out = clip_module(clip_text, clip_image)

    check("Consistency score shape", clip_out["consistency_score"].shape == (batch_size,),
          f"got {clip_out['consistency_score'].shape}")
    check("Consistency score range",
          (clip_out["consistency_score"] >= -1).all() and (clip_out["consistency_score"] <= 1).all(),
          f"range [{clip_out['consistency_score'].min():.3f}, {clip_out['consistency_score'].max():.3f}]")

    # ── 5. Fusion Layer ──────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("5. Fusion Layer (Early Fusion)")
    fusion = DummyFusionLayer()
    fusion_out = fusion(
        text_feat=text_out["features"],
        vision_feat=vision_out["features"],
        forensic_feat=forensic_out["features"],
        consistency_score=clip_out["consistency_score"],
    )

    check("Fusion logits shape", fusion_out["logits"].shape == (batch_size, 4),
          f"got {fusion_out['logits'].shape}")
    check("Fused features shape", fusion_out["fused_features"].shape == (batch_size, 1665),
          f"got {fusion_out['fused_features'].shape}")

    # ── 6. End-to-End Flow ───────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("6. End-to-End Data Flow")

    # Simulate: raw text + image → final prediction
    final_probs = torch.softmax(fusion_out["logits"], dim=-1)
    final_preds = torch.argmax(final_probs, dim=-1)
    label_map = {0: "Likely Authentic", 1: "Potentially Misleading",
                 2: "Potentially Manipulated", 3: "Multimodal Fake"}

    check("Final predictions valid", 
          all(p.item() in label_map for p in final_preds),
          f"predictions: {[label_map[p.item()] for p in final_preds]}")

    check("Probabilities sum to 1",
          torch.allclose(final_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5))

    # ── 7. Gradient Flow ─────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("7. Gradient Flow Check")
    target = torch.randint(0, 4, (batch_size,))
    loss = nn.CrossEntropyLoss()(fusion_out["logits"], target)
    loss.backward()

    has_grads = any(p.grad is not None for p in fusion.parameters())
    check("Gradients flow through fusion", has_grads)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("  All interface checks PASSED! Ready to build real models.")
    else:
        print("  Some checks FAILED! Fix before proceeding.")
    print(f"{'='*60}\n")

    return checks_passed == total_checks


if __name__ == "__main__":
    success = run_skeleton()
    sys.exit(0 if success else 1)
