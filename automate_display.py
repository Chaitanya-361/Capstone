"""
automate_display.py

Automated Presentation & Demo Suite for Multimodal Fake News & Deepfake Detection.
Automatically steps through all benchmark demo cases, executes the inference pipeline,
generates a rich visual HTML report, and opens it automatically in your browser.
"""

import os
import sys
import time
import glob
import webbrowser
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fusion.model import MultimodalFakeNewsDetector
from vision_branch.image_analyzer import ImageAnalyzer


def load_demo_cases():
    demo_dir = os.path.join(BASE_DIR, "demo_samples")
    folders = sorted([d for d in os.listdir(demo_dir) if os.path.isdir(os.path.join(demo_dir, d)) and d.startswith("0")])
    cases = []

    for folder in folders:
        folder_path = os.path.join(demo_dir, folder)
        headline_file = os.path.join(folder_path, "headline.txt")
        info_file = os.path.join(folder_path, "info.txt")

        # Find image
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not images or not os.path.exists(headline_file):
            continue

        with open(headline_file, "r", encoding="utf-8") as f:
            headline = f.read().strip()

        info = ""
        if os.path.exists(info_file):
            with open(info_file, "r", encoding="utf-8") as f:
                info = f.read().strip()

        img_path = os.path.join(folder_path, images[0])

        cases.append({
            "name": folder.replace("_", " "),
            "headline": headline,
            "info": info,
            "image_path": img_path,
            "image_filename": images[0]
        })

    return cases


def run_automated_display():
    print("\n" + "=" * 70)
    print("      AUTOMATED MULTIMODAL DEMONSTRATION & DISPLAY SUITE")
    print("=" * 70)
    print("Initializing models on device (CPU/GPU)...")

    detector = MultimodalFakeNewsDetector()
    image_analyzer = ImageAnalyzer()

    cases = load_demo_cases()
    print(f"Loaded {len(cases)} pre-configured demo cases from demo_samples/.\n")

    results_html_cards = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] Executing Demo: {case['name']}")
        print(f"  Claim Text: \"{case['headline']}\"")
        print(f"  Visual Asset: {os.path.basename(case['image_path'])}")

        start_time = time.time()
        
        # 1. Image Spatial & Frequency Analysis
        img_res = image_analyzer.analyze(case["image_path"])

        # 2. Joint Multimodal Fusion
        fusion_res = detector.predict(text=case["headline"], image=case["image_path"], fusion_strategy="cross_attention")
        elapsed = time.time() - start_time

        verdict = fusion_res["verdict"]
        conf = fusion_res["confidence"]

        # Color mapping
        color_map = {
            "Likely Authentic": ("#10b981", "✅"),
            "Potentially Misleading": ("#f59e0b", "⚠️"),
            "Potentially Manipulated": ("#ef4444", "🚨"),
            "Multimodal Fake Content": ("#f43f5e", "🛑")
        }
        border_color, badge = color_map.get(verdict, ("#3b82f6", "ℹ️"))

        print(f"  --> Verdict: {badge} {verdict} (Confidence: {conf:.1%}) in {elapsed:.2f}s")
        print(f"      [Vision Deepfake: {img_res['spatial_deepfake_prob']:.1%} | FFT Forensic: {img_res['frequency_synthetic_score']:.1%}]")

        # Save FFT heatmap and annotated face image
        out_heatmap_name = f"fft_heatmap_{i}.png"
        out_heatmap_path = os.path.join(BASE_DIR, "results", out_heatmap_name)
        os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
        img_res["spectrum_image"].save(out_heatmap_path)

        out_annotated_name = f"annotated_face_{i}.png"
        out_annotated_path = os.path.join(BASE_DIR, "results", out_annotated_name)
        img_res["annotated_image"].save(out_annotated_path)

        # Copy original image for HTML display
        rel_orig = os.path.relpath(case["image_path"], os.path.join(BASE_DIR, "results")).replace("\\", "/")

        explanation_text = fusion_res.get("explanation", {}).get("summary", "") if isinstance(fusion_res.get("explanation"), dict) else str(fusion_res.get("explanation", ""))

        card_html = f"""
        <div class="case-card" style="border-left: 8px solid {border_color};">
            <div class="case-header">
                <h2>{case['name']}</h2>
                <span class="badge" style="background-color: {border_color};">{badge} {verdict} ({conf:.1%})</span>
            </div>
            <p><strong>Claim:</strong> <em>"{case['headline']}"</em></p>
            <p class="info-text">{case['info']}</p>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="val">{img_res['spatial_deepfake_prob']:.1%}</span>
                    <span class="lbl">Spatial Deepfake (EfficientNet)</span>
                </div>
                <div class="metric-box">
                    <span class="val">{img_res['frequency_synthetic_score']:.1%}</span>
                    <span class="lbl">2D-FFT Forensic Score</span>
                </div>
                <div class="metric-box">
                    <span class="val">{img_res['face_detected']}</span>
                    <span class="lbl">Face Detected ({img_res['num_faces']})</span>
                </div>
                <div class="metric-box">
                    <span class="val">{elapsed:.2f}s</span>
                    <span class="lbl">Inference Latency</span>
                </div>
            </div>
            <div class="image-row">
                <div class="img-container">
                    <h4>Input Image</h4>
                    <img src="{rel_orig}" alt="Input Image" />
                </div>
                <div class="img-container">
                    <h4>Face Detection</h4>
                    <img src="{out_annotated_name}" alt="Face Detection" />
                </div>
                <div class="img-container">
                    <h4>2D-FFT Power Spectrum Heatmap</h4>
                    <img src="{out_heatmap_name}" alt="FFT Spectrum" />
                </div>
            </div>
            <div class="explanation-box">
                <strong>Explainable AI Synthesis:</strong> {explanation_text}
            </div>
        </div>
        """
        results_html_cards.append(card_html)

    # Build Complete HTML Dashboard
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Automated Demo Display — Multimodal Fake News & Deepfake Detection</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b0f19;
            color: #f1f5f9;
            margin: 0;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #1e1b4b, #0f172a);
            border-radius: 12px;
            margin-bottom: 2.5rem;
            border: 1px solid #334155;
        }}
        .header h1 {{ margin: 0; font-size: 2.4rem; color: #ffffff; }}
        .header p {{ color: #94a3b8; font-size: 1.1rem; margin-top: 0.6rem; }}
        .case-card {{
            background-color: #1e293b;
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            border-top: 1px solid #334155;
            border-right: 1px solid #334155;
            border-bottom: 1px solid #334155;
        }}
        .case-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 0.8rem;
            margin-bottom: 1rem;
        }}
        .case-header h2 {{ margin: 0; font-size: 1.5rem; }}
        .badge {{
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-weight: 600;
            color: #ffffff;
            font-size: 0.95rem;
        }}
        .info-text {{ color: #94a3b8; font-size: 0.95rem; line-height: 1.5; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1.2rem 0;
        }}
        .metric-box {{
            background-color: #0f172a;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .metric-box .val {{ display: block; font-size: 1.4rem; font-weight: bold; color: #38bdf8; }}
        .metric-box .lbl {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; display: block; }}
        .image-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        .img-container {{
            background-color: #0f172a;
            padding: 0.8rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .img-container h4 {{ margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #cbd5e1; }}
        .img-container img {{
            max-width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 6px;
        }}
        .explanation-box {{
            background-color: #0f172a;
            border-left: 4px solid #38bdf8;
            padding: 1rem;
            border-radius: 4px;
            font-size: 0.95rem;
            color: #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Multimodal Fake News & Deepfake Detection</h1>
        <p>Automated Capstone Live Demonstration Display — VIT-AP University</p>
    </div>
    <div class="container">
        {''.join(results_html_cards)}
    </div>
</body>
</html>
"""

    report_path = os.path.join(BASE_DIR, "results", "demo_display_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] All {len(cases)} demo cases successfully analyzed!")
    print(f"  Generated Automated Display Dashboard: {report_path}")
    print("=" * 70)

    try:
        webbrowser.open(report_path)
        print("  Opened live presentation dashboard in your browser!\n")
    except Exception:
        pass


if __name__ == "__main__":
    run_automated_display()
