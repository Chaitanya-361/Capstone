"""Text branch package for multimodal fake news detection."""

from text_branch.config import TextBranchConfig, config
from text_branch.model import PretrainedTextBranch
from text_branch.feature_extractor import TextFeatureExtractor

__all__ = ["TextBranchConfig", "config", "PretrainedTextBranch", "TextFeatureExtractor"]
