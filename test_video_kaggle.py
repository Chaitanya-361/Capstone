import sys
sys.path.insert(0, r"C:\Users\shaik\Desktop\Capstone-main")
from vision_branch.image_analyzer import VisionBranch, ForensicBranch
from video_branch.video_analyzer import VideoAnalyzer

class LightweightDetector:
    def __init__(self, v, f):
        self.vision_branch = v
        self.forensic_branch = f

v = VisionBranch()
f = ForensicBranch()
detector = LightweightDetector(v, f)
analyzer = VideoAnalyzer(detector=detector, num_frames=8)

for path, name in [
    (r"C:\Users\shaik\Desktop\Capstone-main\data\video_eval\real\kaggle_real_1.mp4", "Kaggle Real 1"),
    (r"C:\Users\shaik\Desktop\Capstone-main\data\video_eval\fake\kaggle_fake_1.mp4", "Kaggle Fake 1"),
    (r"C:\Users\shaik\Desktop\Capstone-main\data\video_eval\real\kaggle_real_2.mp4", "Kaggle Real 2"),
    (r"C:\Users\shaik\Desktop\Capstone-main\data\video_eval\fake\kaggle_fake_2.mp4", "Kaggle Fake 2"),
]:
    res = analyzer.analyze(path)
    v_meta = res["video"]
    print(f"=== {name} ===")
    print(f"   Video Risk:        {v_meta['video_risk']:.4f}")
    print(f"   Temporal Jitter:   {v_meta['temporal_jitter']:.4f}")
    print(f"   Flagged Ratio:     {v_meta['flagged_frame_ratio']:.2%}")
    print(f"   Is Flagged:        {v_meta['is_flagged']}")
