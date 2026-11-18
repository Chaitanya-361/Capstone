"""Multimodal fusion package implementing early, late, and cross-attention fusion."""

from fusion.cross_attention import MultiHeadCrossAttentionBlock, CrossAttentionFusion
from fusion.consistency import CrossModalConsistency
from fusion.model import EarlyFusion, LateFusion, MultimodalFakeNewsDetector

__all__ = [
    "MultiHeadCrossAttentionBlock",
    "CrossAttentionFusion",
    "CrossModalConsistency",
    "EarlyFusion",
    "LateFusion",
    "MultimodalFakeNewsDetector",
]
