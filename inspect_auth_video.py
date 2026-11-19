from vision_branch.model import VisionBranch
from forensic_branch.fft_detector import ForensicBranch
from video_branch.video_analyzer import VideoAnalyzer

class FastDet:
    def __init__(self):
        self.vision_branch = VisionBranch(device="cpu")
        self.forensic_branch = ForensicBranch()

det = FastDet()
analyzer = VideoAnalyzer(detector=det, num_frames=5)
frames = analyzer.extract_frames("demo_samples/06_Video_Deepfake_Demo/authentic_interview_sample.mp4")
for f in frames:
    face_pil, face_detected = analyzer.detect_and_crop_face(f["image_bgr"], f["image_rgb"])
    risk, v_score, f_score = analyzer.analyze_frame(face_pil, f["pil_image"])
    idx = f["frame_idx"]
    print(f"Frame {idx}: face_detected={face_detected}, v_score={v_score:.3f}, f_score={f_score:.3f}")
