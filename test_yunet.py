
import sys, cv2, os
sys.path.insert(0, r'C:\Users\shaik\Desktop\Capstone-main')
from vision_branch.model import VisionBranch
from PIL import Image

v = VisionBranch()
model_path = r'C:\Users\shaik\Desktop\Capstone-main\face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN_create(model_path, '', (320, 320), 0.6, 0.3, 5000)

for path, name in [
    (r'C:\Users\shaik\Desktop\Capstone-main\data\video_eval\real\kaggle_real_1.mp4', 'Kaggle Real 1'),
    (r'C:\Users\shaik\Desktop\Capstone-main\data\video_eval\fake\kaggle_fake_1.mp4', 'Kaggle Fake 1'),
    (r'C:\Users\shaik\Desktop\Capstone-main\data\video_eval\real\kaggle_real_2.mp4', 'Kaggle Real 2'),
    (r'C:\Users\shaik\Desktop\Capstone-main\data\video_eval\fake\kaggle_fake_2.mp4', 'Kaggle Fake 2'),
]:
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        h, w, _ = frame.shape
        detector.setInputSize((w, h))
        faces = detector.detect(frame)
        if faces[1] is not None:
            x, y, fw, fh = faces[1][0][0:4].astype(int)
            mw, mh = int(0.2 * fw), int(0.2 * fh)
            x1, y1 = max(0, x - mw), max(0, y - mh)
            x2, y2 = min(w, x + fw + mw), min(h, y + fh + mh)
            face_crop = frame[y1:y2, x1:x2]
            pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            res = v.analyze(pil_face)
            p_fake = res['deepfake_prob']
            lbl = res['label']
            print(f'{name}: face=[{x1},{y1},{x2},{y2}] -> Prob Fake={p_fake:.4f}, Label={lbl}')
        else:
            print(f'{name}: No face detected')
