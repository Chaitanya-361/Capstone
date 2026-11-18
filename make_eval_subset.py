"""
make_eval_subset.py

Copies 1,000 real and 1,000 fake images from the 140k Real & Fake Faces dataset
into data/image_eval/real and data/image_eval/fake for benchmark evaluation.
"""

import os
import glob
import random
import shutil

SRC = "data/raw_140k/real_vs_fake/real-vs-fake/test"
DST = "data/image_eval"
N = 1000

random.seed(42)

for cls in ["real", "fake"]:
    src_dir = os.path.join(SRC, cls)
    dst_dir = os.path.join(DST, cls)
    files = glob.glob(os.path.join(src_dir, "*.jpg"))
    os.makedirs(dst_dir, exist_ok=True)
    sample_files = random.sample(files, min(N, len(files)))
    for f in sample_files:
        shutil.copy(f, dst_dir)
    print(f"[{cls}] Successfully copied {len(sample_files)} images to {dst_dir}")
