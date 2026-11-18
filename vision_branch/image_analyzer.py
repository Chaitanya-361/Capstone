"""
image_analyzer.py

Standalone and unified Image Analysis Engine for Deepfake & Synthetic Image Detection.
"""

import os
import argparse
from typing import Dict, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torch

from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch


class ImageAnalyzer:
    """
    Unified Image Deepfake & Manipulation Analyzer.
    Coordinates face detection, spatial CNN inference, and frequency spectrum analysis.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_branch = VisionBranch(device=self.device)
        self.forensic_branch = ForensicBranch()

        self.yunet_detector = None
        self.face_cascade = None

        yunet_candidates = [
            os.path.join(os.path.dirname(__file__), "..", "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx"),
            "face_detection_yunet_2023mar.onnx",
        ]
        yunet_path = next((p for p in yunet_candidates if os.path.exists(p)), None)
        if yunet_path and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
            except Exception:
                self.yunet_detector = None

        try:
            if hasattr(cv2, "CascadeClassifier"):
                cascade_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
                cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
                if not os.path.exists(cascade_path):
                    cascade_path = "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def detect_and_crop_faces(self, image_pil: Image.Image) -> Dict:
        rgb_img = np.array(image_pil.convert("RGB"))
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        h, w, _ = rgb_img.shape
        annotated = rgb_img.copy()

        # 1. Try YuNet face detector
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                _, faces = self.yunet_detector.detect(bgr_img)
                if faces is not None and len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, fw, fh = largest_face[0:4].astype(int)
                    margin_w = int(0.20 * fw)
                    margin_h = int(0.20 * fh)
                    x1 = max(0, x - margin_w)
                    y1 = max(0, y - margin_h)
                    x2 = min(w, x + fw + margin_w)
                    y2 = min(h, y + fh + margin_h)

                    face_crop = rgb_img[y1:y2, x1:x2]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated, f"Face: {fw}x{fh}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return {
                        "cropped_face": Image.fromarray(face_crop),
                        "face_detected": True,
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "annotated_image": Image.fromarray(annotated),
                        "num_faces": len(faces)
                    }
            except Exception:
                pass

        # 2. Try Haar Cascade fallback
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, fw, fh = largest_face
                    margin_w = int(0.20 * fw)
                    margin_h = int(0.20 * fh)
                    x1 = max(0, x - margin_w)
                    y1 = max(0, y - margin_h)
                    x2 = min(w, x + fw + margin_w)
                    y2 = min(h, y + fh + margin_h)

                    face_crop = rgb_img[y1:y2, x1:x2]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated, f"Face: {fw}x{fh}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return {
                        "cropped_face": Image.fromarray(face_crop),
                        "face_detected": True,
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "annotated_image": Image.fromarray(annotated),
                        "num_faces": len(faces)
                    }
            except Exception:
                pass

        return {
            "cropped_face": image_pil,
            "face_detected": False,
            "bbox": None,
            "annotated_image": Image.fromarray(annotated),
            "num_faces": 0
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        face_info = self.detect_and_crop_faces(image)
        eval_face = face_info["cropped_face"]

        vision_res = self.vision_branch.analyze(eval_face)
        spatial_risk = float(vision_res["deepfake_prob"])

        forensic_res = self.forensic_branch.analyze(image)
        frequency_risk = float(forensic_res["synthetic_score"])

        if face_info["face_detected"]:
            combined_risk = max(spatial_risk, 0.8 * frequency_risk)
        else:
            combined_risk = max(frequency_risk, 0.5 * spatial_risk)

        if combined_risk > 0.65:
            verdict = "High Risk: Deepfake / Manipulated"
        elif combined_risk > 0.45:
            verdict = "Suspicious: Potential Artifacts Detected"
        else:
            verdict = "Low Risk: Likely Authentic"

        return {
            "verdict": verdict,
            "combined_risk": float(combined_risk),
            "spatial_deepfake_prob": float(spatial_risk),
            "frequency_synthetic_score": float(frequency_risk),
            "high_freq_energy": float(forensic_res.get("high_freq_energy", 0.0)),
            "face_detected": face_info["face_detected"],
            "num_faces": face_info["num_faces"],
            "bbox": face_info["bbox"],
            "annotated_image": face_info["annotated_image"],
            "cropped_face": face_info["cropped_face"],
            "spectrum_image": forensic_res["spectrum_image"],
            "vision_features": vision_res["features"],
            "forensic_features": forensic_res["features"]
        }
