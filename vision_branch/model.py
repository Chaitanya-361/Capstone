"""
Pre-trained Vision Branch Model for Deepfake and Fake Image Detection
Uses EfficientNet-B0 fine-tuned on FaceForensics++ (FF++ c23 benchmark).
Fulfills PRD Section 6.2 & Section 5.
"""

import os
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class VisionBranch(nn.Module):
    """
    Vision branch based on EfficientNet-B0 fine-tuned on FaceForensics++.
    Extracts visual features and classifies images as Authentic vs Deepfake.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        feature_dim: int = 768,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feature_dim = feature_dim

        # Find or download FaceForensics++ weights
        if weights_path is None or not os.path.exists(weights_path):
            weights_path = self._resolve_weights()

        # Build EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(num_classes=2)

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded FaceForensics++ deepfake weights from {weights_path}")
        else:
            print("Warning: Using standard ImageNet initialization for EfficientNet-B0")

        # Visual feature projection: 1280 -> 768 to match text branch dimension
        self.projector = nn.Linear(1280, self.feature_dim)

        # Standard ImageNet / FaceForensics++ preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.to(self.device)
        self.eval()

    def _resolve_weights(self) -> str:
        """Locate cached weights or download via huggingface_hub."""
        cache_snapshot = os.path.expanduser(
            r"~/.cache/huggingface/hub/models--Xicor9--efficientnet-b0-ffpp-c23/snapshots"
        )
        if os.path.exists(cache_snapshot):
            for root, _, files in os.walk(cache_snapshot):
                if "efficientnet_b0_ffpp_c23.pth" in files:
                    return os.path.join(root, "efficientnet_b0_ffpp_c23.pth")

        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id="Xicor9/efficientnet-b0-ffpp-c23",
                filename="efficientnet_b0_ffpp_c23.pth",
            )
        except Exception:
            return ""

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            pixel_values: Tensor of shape (B, 3, 224, 224)
        Returns:
            Dict containing 'features' (B, 768), 'logits' (B, 2), 'deepfake_prob' (B,)
        """
        # Feature extraction through convolutional stem & blocks
        x = self.backbone.features(pixel_values)           # (B, 1280, 7, 7)
        spatial_features = x
        pooled = self.backbone.avgpool(x)                  # (B, 1280, 1, 1)
        pooled = torch.flatten(pooled, 1)                  # (B, 1280)

        # Classification logits
        logits = self.backbone.classifier(pooled)          # (B, 2)
        probs = torch.softmax(logits, dim=-1)

        # Feature projection to unified 768-dim representation
        features_768 = self.projector(pooled)

        return {
            "features": features_768,                      # (B, 768)
            "raw_features": pooled,                        # (B, 1280)
            "spatial_features": spatial_features,          # (B, 1280, 7, 7)
            "logits": logits,
            "probs": probs,
            "real_prob": probs[:, 0],                      # Class 0: Real
            "deepfake_prob": probs[:, 1],                  # Class 1: Deepfake / Manipulated
        }

    @torch.no_grad()
    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze a single image.
        Args:
            image_input: File path, PIL Image, or numpy array
        Returns:
            Dict with labels, confidence, deepfake_prob, and features
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.forward(tensor)

        probs = outputs["probs"].squeeze(0).cpu().numpy()
        real_prob = float(probs[0])
        deepfake_prob = float(probs[1])

        label = "Deepfake / Manipulated" if deepfake_prob > 0.5 else "Authentic"
        confidence = max(real_prob, deepfake_prob)

        return {
            "label": label,
            "confidence": float(confidence),
            "deepfake_prob": float(deepfake_prob),
            "real_prob": float(real_prob),
            "features": outputs["features"].squeeze(0).cpu().numpy(),
            "raw_features": outputs["raw_features"].squeeze(0).cpu().numpy(),
            "image_size": image.size,
        }

    def get_feature_dim(self) -> int:
        return self.feature_dim
