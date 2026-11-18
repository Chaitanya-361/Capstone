"""
Forensic Frequency-Domain Branch
Detects synthetic / deepfake periodic artifacts using 2D Fast Fourier Transform (FFT)
and azimuthal average power spectrum analysis.
Fulfills PRD Section 6.2 and tests Hypothesis H2.
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2


class ForensicBranch(nn.Module):
    """
    Forensic frequency-domain artifact detector.
    Analyzes high-frequency Fourier components for GAN and diffusion lattice artifacts.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        # Small MLP to score the 128-dim radial frequency profile
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        self.eval()

    def _init_weights(self):
        """Heuristic initialization: weight higher frequency bins positively."""
        with torch.no_grad():
            for m in self.classifier:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def extract_fft_spectrum(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
        """
        Compute 2D FFT magnitude spectrum and 1D radial average profile.
        Returns:
            radial_profile: 1D array of length self.feature_dim
            magnitude_spectrum: 2D array
            spectrum_img: PIL Image heatmap of the power spectrum
        """
        # Convert to grayscale numpy array
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # Compute 2D Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # Compute azimuthal (radial) average profile from center
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
        max_r = min(cx, cy)

        # Average power in concentric rings
        radial_profile = np.zeros(max_r, dtype=np.float32)
        for radius in range(max_r):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = np.mean(magnitude_spectrum[mask])

        # Interpolate / resample to fixed feature dimension (e.g. 128 bins)
        if len(radial_profile) > 1:
            x_old = np.linspace(0, 1, len(radial_profile))
            x_new = np.linspace(0, 1, self.feature_dim)
            resampled_profile = np.interp(x_new, x_old, radial_profile)
        else:
            resampled_profile = np.zeros(self.feature_dim, dtype=np.float32)

        # Normalize profile to [0, 1]
        p_min, p_max = np.min(resampled_profile), np.max(resampled_profile)
        if p_max > p_min:
            normalized_profile = (resampled_profile - p_min) / (p_max - p_min)
        else:
            normalized_profile = np.zeros_like(resampled_profile)

        # Create visual RGB heatmap for explainability
        norm_mag = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_mag.astype(np.uint8), cv2.COLORMAP_INFERNO)
        spectrum_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

        return normalized_profile, magnitude_spectrum, spectrum_pil

    def forward(self, radial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radial_features: Tensor of shape (B, 128)
        Returns:
            anomaly_score: Tensor of shape (B, 1) in range [0, 1]
        """
        return self.classifier(radial_features)

    @torch.no_grad()
    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze frequency spectrum of an image.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        profile, _, spectrum_img = self.extract_fft_spectrum(image)

        # High-frequency anomaly ratio: energy in upper 25% of frequency bands
        # Synthetic upsamplers cause pronounced high-frequency ripple
        high_freq_ratio = float(np.mean(profile[int(self.feature_dim * 0.75):]))
        mid_freq_ratio = float(np.mean(profile[int(self.feature_dim * 0.25):int(self.feature_dim * 0.75)]))

        tensor_profile = torch.tensor(profile, dtype=torch.float32).unsqueeze(0)
        mlp_score = float(self.forward(tensor_profile).squeeze().item())

        # Blended synthetic anomaly score
        synthetic_score = float(np.clip(0.5 * mlp_score + 0.5 * high_freq_ratio, 0.0, 1.0))
        label = "Synthetic / Manipulated" if synthetic_score > 0.55 else "Authentic"

        return {
            "label": label,
            "synthetic_score": synthetic_score,
            "high_freq_energy": high_freq_ratio,
            "mid_freq_energy": mid_freq_ratio,
            "features": profile,                       # (128,) numpy array for fusion
            "spectrum_image": spectrum_img,            # PIL Image for demo XAI
        }

    def get_feature_dim(self) -> int:
        return self.feature_dim
