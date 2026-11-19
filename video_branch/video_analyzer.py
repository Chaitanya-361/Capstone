"""
Video Analysis Branch for Multimodal Deepfake and Fake News Detection.
Integrates temporal frame sampling, facial cropping, forensic scoring,
and cross-modal fake news detection.
"""

import os
import argparse
import tempfile
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torch


class VideoAnalyzer:
    """
    Performs multi-frame video analysis for deepfake manipulation detection:
    1. Uniformly samples N frames across the video using OpenCV.
    2. Detects and crops faces using Haar Cascade (20% margin, fallback to full frame).
    3. Analyzes spatial deepfake probability (VisionBranch) and frequency artifacts (ForensicBranch).
    4. Aggregates:
       - Video risk = Top-3 frame risk average.
       - Flagged frame ratio = Frames exceeding risk threshold (>= 0.55).
       - Temporal jitter = Standard deviation of frame-to-frame deepfake probabilities.
       - Key frame = Most suspicious frame (argmax of risk).
    5. Integrates with MultimodalFakeNewsDetector for joint text-video classification.
    """

    def __init__(self, detector=None, num_frames: int = 16, risk_threshold: float = 0.55):
        self.detector = detector
        self.num_frames = num_frames
        self.risk_threshold = risk_threshold

        # Initialize face detectors: YuNet (Deep Learning) or Haar Cascade
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

    def extract_frames(self, video_path: str) -> List[Dict]:
        """Uniformly samples frames across the video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video contains 0 frames or invalid video stream: {video_path}")

        sample_indices = np.linspace(0, total_frames - 1, min(self.num_frames, total_frames), dtype=int)

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_sec = float(idx) / float(fps)
                frames.append({
                    "frame_idx": int(idx),
                    "timestamp_sec": timestamp_sec,
                    "image_bgr": frame,
                    "image_rgb": rgb_frame,
                    "pil_image": Image.fromarray(rgb_frame),
                })
        cap.release()
        return frames

    def detect_and_crop_face(self, bgr_frame: np.ndarray, rgb_frame: np.ndarray) -> (Image.Image, bool):
        """Detects the largest face with 20% margin, or returns full frame if no face is found."""
        h, w, _ = bgr_frame.shape

        # 1. Try YuNet deep learning face detector
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                _, faces = self.yunet_detector.detect(bgr_frame)
                if faces is not None and len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, fw, fh = largest_face[0:4].astype(int)
                    margin_w = int(0.20 * fw)
                    margin_h = int(0.20 * fh)
                    x1 = max(0, x - margin_w)
                    y1 = max(0, y - margin_h)
                    x2 = min(w, x + fw + margin_w)
                    y2 = min(h, y + fh + margin_h)
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    return Image.fromarray(face_crop), True
            except Exception:
                pass

        # 2. Try Haar Cascade fallback
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
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
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    return Image.fromarray(face_crop), True
            except Exception:
                pass

        return Image.fromarray(rgb_frame), False

    def analyze_frame(self, face_pil: Image.Image, full_frame_pil: Image.Image) -> (float, float, float):
        """Computes risk scores using vision and forensic branches."""
        if self.detector is not None:
            # Spatial vision branch on face crop
            v_res = self.detector.vision_branch.analyze(face_pil)
            v_score = float(v_res.get("deepfake_prob", 0.0))

            # Frequency forensic branch on full frame
            f_res = self.detector.forensic_branch.analyze(full_frame_pil)
            f_score = float(f_res.get("synthetic_score", 0.0))

            frame_risk = max(v_score, f_score)
            return frame_risk, v_score, f_score

        return 0.0, 0.0, 0.0

    def analyze(self, video_path: str, text: Optional[str] = None, fusion_strategy: str = "cross_attention") -> Dict:
        """Runs full temporal video analysis and joins with text claim if provided."""
        frames = self.extract_frames(video_path)
        if not frames:
            raise RuntimeError(f"No frames could be extracted from {video_path}")

        frame_records = []
        deepfake_probs = []

        for f in frames:
            face_pil, face_detected = self.detect_and_crop_face(f["image_bgr"], f["image_rgb"])
            risk, v_score, f_score = self.analyze_frame(face_pil, f["pil_image"])

            frame_records.append({
                "frame_idx": f["frame_idx"],
                "timestamp_sec": round(f["timestamp_sec"], 2),
                "risk": float(risk),
                "vision_score": float(v_score),
                "forensic_score": float(f_score),
                "face_detected": face_detected,
                "pil_image": f["pil_image"],
                "face_pil": face_pil,
            })
            deepfake_probs.append(v_score)

        # 1. Top-k mean (k=3)
        sorted_risks = sorted([r["risk"] for r in frame_records], reverse=True)
        top_3_risks = sorted_risks[:min(3, len(sorted_risks))]
        video_risk = float(np.mean(top_3_risks)) if top_3_risks else 0.0

        # 2. Flagged frame ratio
        flagged_frames = [r for r in frame_records if r["risk"] >= self.risk_threshold]
        flagged_ratio = float(len(flagged_frames)) / float(len(frame_records))

        # 3. Temporal jitter
        if len(deepfake_probs) > 1:
            diffs = np.diff(deepfake_probs)
            temporal_jitter = float(np.std(diffs))
        else:
            temporal_jitter = 0.0

        # 4. Key frame selection (argmax risk)
        key_record = max(frame_records, key=lambda r: r["risk"])
        key_frame = key_record["pil_image"]
        key_frame_time = key_record["timestamp_sec"]

        video_metadata = {
            "video_risk": video_risk,
            "flagged_frame_ratio": flagged_ratio,
            "temporal_jitter": temporal_jitter,
            "key_frame_time_sec": key_frame_time,
            "is_flagged": bool(flagged_ratio >= 0.40 or (video_risk >= 0.60 and temporal_jitter >= 0.15)),
            "total_frames_sampled": len(frame_records),
            "per_frame": [
                {
                    "frame_idx": r["frame_idx"],
                    "timestamp_sec": r["timestamp_sec"],
                    "risk": round(r["risk"], 4),
                    "face_detected": r["face_detected"],
                }
                for r in frame_records
            ]
        }

        # 5. Multimodal Reasoning if text is provided
        if self.detector is not None and text:
            result = self.detector.predict(text=text, image=key_frame, fusion_strategy=fusion_strategy)
            result["key_frame"] = key_frame
            result["video"] = video_metadata

            orig_exp = result.get("explanation", {}).get("summary", "") if isinstance(result.get("explanation"), dict) else result.get("explanation", "")
            video_exp = (
                f" Video analysis evaluated {len(frame_records)} frames "
                f"(video risk: {video_risk:.1%}, flagged frames: {flagged_ratio:.0%}, "
                f"temporal jitter: {temporal_jitter:.3f}). "
                f"Most suspicious key frame observed at t={key_frame_time:.1f}s."
            )
            if isinstance(result.get("explanation"), dict):
                result["explanation"]["summary"] = orig_exp + " " + video_exp
            else:
                result["explanation"] = orig_exp + " " + video_exp
            return result
        else:
            return {
                "key_frame": key_frame,
                "video": video_metadata,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Video Analyzer on a video file.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--text", default="", help="Headline or claim text")
    parser.add_argument("--fusion", default="cross_attention", help="Fusion strategy")
    parser.add_argument("--frames", type=int, default=16, help="Frames to sample")
    args = parser.parse_args()

    try:
        from fusion.model import MultimodalFakeNewsDetector
        detector = MultimodalFakeNewsDetector()
    except Exception as e:
        print(f"Running without full fusion detector ({e})")
        detector = None

    analyzer = VideoAnalyzer(detector=detector, num_frames=args.frames)
    res = analyzer.analyze(args.video, text=args.text, fusion_strategy=args.fusion)

    print("\n================ Video Analysis Results ================")
    print(f"Video Risk Score: {res['video']['video_risk']:.2%}")
    print(f"Flagged Frame Ratio: {res['video']['flagged_frame_ratio']:.1%}")
    print(f"Key Frame Timestamp: {res['video']['key_frame_time_sec']:.2f}s")
    print(f"Temporal Jitter: {res['video']['temporal_jitter']:.4f}")
    if "verdict" in res:
        print(f"Multimodal Verdict: {res['verdict']} ({res.get('confidence', 0.0):.1%})")
