import os
import urllib.request
import time

out_dir = os.path.join(os.path.dirname(__file__), "models", "pretrained_deepfake")
os.makedirs(out_dir, exist_ok=True)

files = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors"
]

base_url = "https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/"

for fname in files:
    target_path = os.path.join(out_dir, fname)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        if fname != "model.safetensors" or os.path.getsize(target_path) > 300_000_000:
            print(f"Already downloaded: {fname} ({os.path.getsize(target_path)} bytes)")
            continue

    url = base_url + fname
    print(f"Downloading {fname} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    start = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as out_f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 4  # 4MB chunks
        last_log = time.time()
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_log > 3:
                last_log = time.time()
                pct = (downloaded / total * 100) if total else 0
                mb = downloaded / (1024 * 1024)
                tot_mb = total / (1024 * 1024)
                print(f"  Progress: {mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)")

    print(f"Finished {fname} in {time.time() - start:.1f}s ({os.path.getsize(target_path)} bytes)")

print("\nAll files downloaded successfully to:", out_dir)
