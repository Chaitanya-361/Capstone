"""
Text Branch Configuration
Hyperparameters and model settings for the RoBERTa-based text branch.
"""

from dataclasses import dataclass
import os


@dataclass
class TextBranchConfig:
    """Configuration for pre-trained RoBERTa text feature extraction and classification."""
    model_name: str = "roberta-base"
    max_seq_length: int = 256
    feature_dim: int = 768
    num_labels: int = 2  # Binary: 0=Real, 1=Fake / Suspicious
    label_names: tuple = ("Real", "Fake")
    device: str = "cpu"


config = TextBranchConfig()
