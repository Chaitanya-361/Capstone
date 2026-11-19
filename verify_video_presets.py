"""
verify_video_presets.py
Verifies VideoAnalyzer on authentic interview vs deepfake video.
"""

import os
from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch
from video_branch.video_analyzer import VideoAnalyzer

class FastDet:
    def __init__(self):
        self.vision_branch = VisionBranch(device="cpu")
        self.forensic_branch = ForensicBranch()

det = FastDet()
analyzer = VideoAnalyzer(detector=det, num_frames=5)

clips = [
    ("demo_samples/06_Video_Deepfake_Demo/authentic_interview_sample.mp4", 0, "Authentic Interview"),
    ("demo_samples/06_Video_Deepfake_Demo/deepfake_speech_sample.mp4", 1, "Deepfake Speech"),
    ("data/video_eval/fake/deepfake_sample_1.mp4", 1, "Deepfake Clip 1"),
    ("data/video_eval/fake/deepfake_video_clip_02.mp4", 1, "Deepfake Clip 2")
]

print("=" * 70)
print("             VIDEO DEEPFAKE DEMO BENCHMARK")
print("=" * 70)

for path, expected, label in clips:
    if os.path.exists(path):
        res = analyzer.analyze(path)["video"]
        risk = res["video_risk"]
        flagged = res["is_flagged"]
        passed = (flagged == bool(expected))
        status = "[PASS]" if passed else "[FAIL]"
        flag_str = "FLAGGED DEEPFAKE" if flagged else "CLEAN / AUTHENTIC"
        print(f"{status} {label:<22} -> Risk: {risk:<6.1%} [{flag_str}]")
print("=" * 70)
