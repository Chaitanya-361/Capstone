"""
Cross-Modal Attention Fusion Strategy
Implements learned multi-head cross-attention between text and image representations,
benchmarked against Early and Late fusion baselines to test Hypothesis H1.
Fulfills PRD Section 3 (H1) and Section 6.4.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttentionBlock(nn.Module):
    """
    Bidirectional multi-head cross-attention block.
    Allows text tokens/features to attend to visual features, and vice versa.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Text attending to Vision
        self.t2v_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.t_norm1 = nn.LayerNorm(embed_dim)
        self.t_norm2 = nn.LayerNorm(embed_dim)
        self.t_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Vision attending to Text
        self.v2t_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.v_norm1 = nn.LayerNorm(embed_dim)
        self.v_norm2 = nn.LayerNorm(embed_dim)
        self.v_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self, text_seq: torch.Tensor, vision_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_seq: (B, L_t, embed_dim)
            vision_seq: (B, L_v, embed_dim)
        Returns:
            attended_text, attended_vision, t2v_weights, v2t_weights
        """
        # Text queries attend to Vision keys & values
        t_attn_out, t2v_weights = self.t2v_attn(
            query=text_seq, key=vision_seq, value=vision_seq
        )
        t_seq = self.t_norm1(text_seq + t_attn_out)
        t_seq = self.t_norm2(t_seq + self.t_ffn(t_seq))

        # Vision queries attend to Text keys & values
        v_attn_out, v2t_weights = self.v2t_attn(
            query=vision_seq, key=text_seq, value=text_seq
        )
        v_seq = self.v_norm1(vision_seq + v_attn_out)
        v_seq = self.v_norm2(v_seq + self.v_ffn(v_seq))

        return t_seq, v_seq, t2v_weights, v2t_weights


class CrossAttentionFusion(nn.Module):
    """
    Primary fusion model per Hypothesis H1.
    Integrates Text, Vision, Forensic Frequency, and Consistency through Cross-Attention.
    """

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        forensic_dim: int = 128,
        shared_dim: int = 512,
        num_classes: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.forensic_dim = forensic_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes

        # Projections into shared attention space
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.vision_proj = nn.Linear(vision_dim, shared_dim)

        # Cross-attention block
        self.cross_attn = MultiHeadCrossAttentionBlock(
            embed_dim=shared_dim, num_heads=num_heads, dropout=dropout
        )

        # Fused classifier
        # Concatenates: attended_text (512) + attended_vision (512) + forensic (128) + consistency (1) = 1153
        total_fused_dim = shared_dim + shared_dim + forensic_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(total_fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

        self._init_weights()
        self.eval()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        forensic_feat: torch.Tensor,
        consistency_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_feat: (B, 768)
            vision_feat: (B, 768)
            forensic_feat: (B, 128)
            consistency_score: (B, 1) or (B,)
        """
        B = text_feat.shape[0]
        text_feat = text_feat.float()
        vision_feat = vision_feat.float()
        forensic_feat = forensic_feat.float()
        consistency_score = consistency_score.float()

        if consistency_score.ndim == 1:
            consistency_score = consistency_score.unsqueeze(-1)

        # Reshape to sequence length 1 for attention: (B, 1, 512)
        t_proj = self.text_proj(text_feat).unsqueeze(1)
        v_proj = self.vision_proj(vision_feat).unsqueeze(1)

        # Cross-attention
        t_att, v_att, t2v_w, v2t_w = self.cross_attn(t_proj, v_proj)

        t_pool = t_att.squeeze(1)  # (B, 512)
        v_pool = v_att.squeeze(1)  # (B, 512)

        # Fuse all streams
        fused = torch.cat([t_pool, v_pool, forensic_feat, consistency_score], dim=-1)

        # Logits
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "fused_features": fused,
            "t2v_attention": t2v_w,
            "v2t_attention": v2t_w,
        }
