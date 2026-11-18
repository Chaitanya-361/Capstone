"""
Cross-Modal Semantic Consistency Module
Measures alignment between text claims and visual content.
Fulfills PRD Section 6.3.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalConsistency(nn.Module):
    """
    Computes cross-modal semantic similarity between text and image embeddings.
    Maps 768-dim text and 768-dim image representations into a shared 512-dim metric space.
    """

    def __init__(self, text_dim: int = 768, vision_dim: int = 768, shared_dim: int = 512):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )

        self._init_projections()
        self.eval()

    def _init_projections(self):
        """Orthogonal / Xavier initialization for shared space."""
        for m in [self.text_proj[0], self.vision_proj[0]]:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            text_features: (B, text_dim)
            vision_features: (B, vision_dim)
        Returns:
            Dict with 'cosine_similarity' (B,), 'consistency_score' (B,)
        """
        # Project to shared multimodal space
        proj_text = F.normalize(self.text_proj(text_features), p=2, dim=-1)
        proj_vision = F.normalize(self.vision_proj(vision_features), p=2, dim=-1)

        # Cosine similarity in range [-1, 1]
        cosine_sim = torch.sum(proj_text * proj_vision, dim=-1)

        # Normalized consistency score in range [0, 1]
        consistency = torch.clamp((cosine_sim + 1.0) / 2.0, 0.0, 1.0)

        return {
            "cosine_similarity": cosine_sim,
            "consistency_score": consistency,
            "projected_text": proj_text,
            "projected_vision": proj_vision,
        }

    @torch.no_grad()
    def compute(self, text_features: np.ndarray, vision_features: np.ndarray) -> Dict:
        """Numpy inference wrapper."""
        t_ten = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0)
        v_ten = torch.tensor(vision_features, dtype=torch.float32).unsqueeze(0)

        out = self.forward(t_ten, v_ten)
        sim = float(out["cosine_similarity"].squeeze().item())
        score = float(out["consistency_score"].squeeze().item())

        verdict = "Semantically Consistent" if score > 0.55 else "Context Mismatch / Low Consistency"

        return {
            "cosine_similarity": sim,
            "consistency_score": score,
            "verdict": verdict,
        }
