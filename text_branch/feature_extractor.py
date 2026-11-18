"""
Text Feature Extractor
High-level interface for extracting text embeddings and predictions.
"""

from typing import Dict, List, Union
import numpy as np
import torch
from text_branch.model import PretrainedTextBranch
from text_branch.config import config


class TextFeatureExtractor:
    """Convenience wrapper for extracting text features and classifications."""

    def __init__(self, device: str = None):
        dev = device or config.device
        self.branch = PretrainedTextBranch(
            model_name=config.model_name,
            feature_dim=config.feature_dim,
            device=dev,
        )

    def extract(self, text: str) -> np.ndarray:
        """Extract 768-dim feature vector."""
        res = self.branch.analyze(text)
        return res["features"]

    def analyze(self, text: str) -> Dict:
        """Full analysis of text."""
        return self.branch.analyze(text)

    def get_feature_dim(self) -> int:
        return self.branch.get_feature_dim()
