"""
create_small_eval_dataset.py
Curates a clean, verified small benchmark dataset for Image and Video Deepfake Accuracy Testing.
"""

import os
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(BASE_DIR, "data", "eval_small_benchmark")

# Structure:
# data/eval_small_benchmark/
#   images/real/
#   images/fake/
#   videos/real/
#   videos/fake/

os.makedirs(os.path.join(BENCH_DIR, "images", "real"), exist_ok=True)
os.makedirs(os.path.join(BENCH_DIR, "images", "fake"), exist_ok=True)
os.makedirs(os.path.join(BENCH_DIR, "videos", "real"), exist_ok=True)
os.makedirs(os.path.join(BENCH_DIR, "videos", "fake"), exist_ok=True)

# 1. Curate Real Images (20 verified samples)
real_sources = [
    os.path.join(BASE_DIR, "demo_samples", "01_Likely_Authentic_Space", "jwst_deep_field.jpg"),
    os.path.join(BASE_DIR, "demo_samples", "02_Likely_Authentic_Politics", "official_portrait.jpg"),
    os.path.join(BASE_DIR, "demo_samples", "03_Potentially_Misleading_Disaster_Hoax", "venice_canal.jpg"),
] + glob.glob(os.path.join(BASE_DIR, "data", "image_eval", "real", "*.jpg"))[:17]

for i, src in enumerate(real_sources, 1):
    if os.path.exists(src):
        dst = os.path.join(BENCH_DIR, "images", "real", f"real_img_{i:02d}.jpg")
        shutil.copy2(src, dst)

# 2. Curate Fake / Deepfake Images (20 verified samples)
fake_sources = [
    os.path.join(BASE_DIR, "demo_samples", "04_Potentially_Manipulated_Deepfake", "stylegan_synthetic_face.jpg"),
    os.path.join(BASE_DIR, "demo_samples", "05_Multimodal_Fake_Conspiracy", "synthetic_lattice_core.jpg"),
] + glob.glob(os.path.join(BASE_DIR, "data", "image_eval", "fake", "*.jpg"))[:18]

for i, src in enumerate(fake_sources, 1):
    if os.path.exists(src):
        dst = os.path.join(BENCH_DIR, "images", "fake", f"fake_img_{i:02d}.jpg")
        shutil.copy2(src, dst)

# 3. Curate Real Videos (4 verified samples)
real_vids = [
    os.path.join(BASE_DIR, "demo_samples", "06_Video_Deepfake_Demo", "authentic_interview_sample.mp4"),
    os.path.join(BASE_DIR, "data", "video_eval", "real", "authentic_sample_1.mp4"),
    os.path.join(BASE_DIR, "data", "video_eval", "real", "real_video_clip_01.mp4"),
    os.path.join(BASE_DIR, "data", "video_eval", "real", "real_video_clip_02.mp4"),
]
for i, src in enumerate(real_vids, 1):
    if os.path.exists(src):
        dst = os.path.join(BENCH_DIR, "videos", "real", f"real_video_{i:02d}.mp4")
        shutil.copy2(src, dst)

# 4. Curate Deepfake Videos (4 verified samples)
fake_vids = [
    os.path.join(BASE_DIR, "demo_samples", "06_Video_Deepfake_Demo", "deepfake_speech_sample.mp4"),
    os.path.join(BASE_DIR, "data", "video_eval", "fake", "deepfake_sample_1.mp4"),
    os.path.join(BASE_DIR, "data", "video_eval", "fake", "deepfake_video_clip_01.mp4"),
    os.path.join(BASE_DIR, "data", "video_eval", "fake", "deepfake_video_clip_02.mp4"),
]
for i, src in enumerate(fake_vids, 1):
    if os.path.exists(src):
        dst = os.path.join(BENCH_DIR, "videos", "fake", f"fake_video_{i:02d}.mp4")
        shutil.copy2(src, dst)

print("Small benchmark dataset successfully created at:")
print(f"  Images: {len(glob.glob(os.path.join(BENCH_DIR, 'images', 'real', '*')))} Real, "
      f"{len(glob.glob(os.path.join(BENCH_DIR, 'images', 'fake', '*')))} Fake")
print(f"  Videos: {len(glob.glob(os.path.join(BENCH_DIR, 'videos', 'real', '*')))} Real, "
      f"{len(glob.glob(os.path.join(BENCH_DIR, 'videos', 'fake', '*')))} Fake")
