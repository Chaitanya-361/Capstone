"""
Explainable AI (XAI) Module
Provides visual, textual, and natural-language explanations for multimodal decisions.
Fulfills PRD Section 6.6 and tests Hypothesis H3.
"""

from typing import Dict, List
import numpy as np
import cv2
from PIL import Image


class MultimodalExplainer:
    """Generates multifaceted explanations across modalities."""

    @staticmethod
    def generate_word_html(word_attributions: List[Dict]) -> str:
        """
        Render text with color-coded attention background highlights.
        Higher attention weights are highlighted in bright orange/amber.
        """
        if not word_attributions:
            return ""

        max_score = max(w["score"] for w in word_attributions) if word_attributions else 1.0
        html_spans = []

        for item in word_attributions:
            norm_score = item["score"] / (max_score + 1e-8)
            alpha = min(0.85, max(0.1, norm_score * 0.75))
            color = f"rgba(245, 158, 11, {alpha:.2f})"
            html_spans.append(
                f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; margin: 1px; color: #f8fafc; font-weight: 500;">{item["token"]}</span>'
            )

        return " ".join(html_spans)

    @staticmethod
    def format_modality_breakdown(scores: Dict[str, float]) -> Dict[str, Dict]:
        """Format scores into structured percentages, status, and color codes."""
        t_fake = scores.get("text_fake_score", 0.0)
        v_deep = scores.get("vision_deepfake_score", 0.0)
        f_synth = scores.get("forensic_synthetic_score", 0.0)
        c_sim = scores.get("consistency_score", 0.5)

        t_credibility = 1.0 - t_fake

        return {
            "Text Credibility": {
                "score": t_credibility,
                "label": "Factual & Credible" if t_fake < 0.40 else ("Sensational / Fake" if t_fake > 0.60 else "Unverified"),
                "color": "#10b981" if t_fake < 0.40 else ("#ef4444" if t_fake > 0.60 else "#f59e0b"),
                "risk": "Low" if t_fake < 0.35 else ("Medium" if t_fake <= 0.60 else "High"),
            },
            "Vision Deepfake": {
                "score": v_deep,
                "label": "Manipulated / Deepfake" if v_deep > 0.55 else "Authentic Image",
                "color": "#ef4444" if v_deep > 0.55 else "#10b981",
                "risk": "High" if v_deep > 0.60 else ("Medium" if v_deep > 0.35 else "Low"),
            },
            "Forensic FFT Spectrum": {
                "score": f_synth,
                "label": "Lattice Artifacts Found" if f_synth > 0.55 else "Natural Spectrum",
                "color": "#ef4444" if f_synth > 0.55 else "#10b981",
                "risk": "High" if f_synth > 0.60 else ("Medium" if f_synth > 0.35 else "Low"),
            },
            "Cross-Modal Consistency": {
                "score": c_sim,
                "label": "Aligned Context" if c_sim > 0.45 else "Context Mismatch",
                "color": "#10b981" if c_sim > 0.45 else "#f59e0b",
                "risk": "Low" if c_sim > 0.45 else ("Medium" if c_sim > 0.35 else "High"),
            },
        }
