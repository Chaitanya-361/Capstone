"""
Multimodal Fake News and Deepfake Detection System Using Deep Learning and Explainable AI
Executive Forensic Intelligence Dashboard

Academic Capstone Project — Department of Computer Science & Engineering
School of Computer Science & Engineering (SCOPE), VIT-AP University

Team Members:
- Shaik Mahammad Usman (23BCE20183) — Video & Image Forensics Subsystem
- Greeshmika Ronanki (23BCE20074) — Vision Branch & Deepfake Detection
- Meegada Nandi (23BCE9614) — Cross-Attention Fusion & Explainable AI (XAI)
- Julakanti Surya Chaitanya Reddy (23BCE8832) — Text Branch & System Integration
"""

import sys
import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Anchor project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fusion.model import MultimodalFakeNewsDetector
from explainability.explainer import MultimodalExplainer
from video_branch.video_analyzer import VideoAnalyzer

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Fake News & Deepfake Detection Platform | VIT-AP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Professional Corporate / Research Lab CSS ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #E2E8F0;
    }

    .stApp {
        background-color: #0B0F17;
        background-image: 
            radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.08) 0, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.08) 0, transparent 50%),
            radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.5) 0, transparent 50%);
    }

    /* Professional Top Navigation Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.6rem;
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nav-logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #2563EB, #4F46E5);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    .nav-titles {
        display: flex;
        flex-direction: column;
    }

    .nav-title-main {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
    }

    .nav-subtitle {
        font-size: 0.76rem;
        color: #94A3B8;
        font-weight: 400;
    }

    .nav-status-group {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nav-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    .nav-badge-green {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .nav-badge-blue {
        background: rgba(37, 99, 235, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(37, 99, 235, 0.25);
    }

    .green-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    /* Section Cards */
    .pro-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    }

    .card-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .card-header-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Executive Verdict Banner */
    .verdict-banner {
        padding: 1.6rem 2rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.4);
    }

    .vb-auth {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(16, 185, 129, 0.4);
    }

    .vb-mislead {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.5) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(245, 158, 11, 0.4);
    }

    .vb-manip {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(239, 68, 68, 0.4);
    }

    .vb-multi {
        background: linear-gradient(135deg, rgba(88, 28, 135, 0.5) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(168, 85, 247, 0.4);
    }

    .verdict-left {
        display: flex;
        flex-direction: column;
    }

    .verdict-tag {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .verdict-name {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        margin: 0;
    }

    .verdict-metrics-row {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .v-metric-item {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }

    .v-metric-label {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .v-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }

    /* 4-Pillar Metric Grid */
    .pillar-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .pillar-box:hover {
        border-color: rgba(59, 130, 246, 0.35);
        transform: translateY(-2px);
    }

    .pillar-box-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .pillar-box-num {
        font-size: 1.95rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.2rem 0;
    }

    .pillar-box-status {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 6px;
        margin-top: 4px;
    }

    /* Professional Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        padding: 0.7rem 1.6rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
        letter-spacing: 0.2px !important;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
        color: #FFFFFF !important;
    }

    /* Clean Input Fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
    }

    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 4px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 6px 14px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(37, 99, 235, 0.2) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(37, 99, 235, 0.4) !important;
    }

    /* Executive Footer */
    .pro-footer {
        text-align: center;
        margin-top: 3.5rem;
        padding: 1.8rem 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.7;
    }

    .pro-footer b {
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loader (Cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    return MultimodalFakeNewsDetector(device="cpu")


# ─── Main Application ───────────────────────────────────────────────────────
def main():
    # ─── Top Executive Navbar ───────────────────────────────────────────────
    st.markdown("""
    <div class="top-navbar">
        <div class="nav-brand">
            <div class="nav-logo-icon">🛡️</div>
            <div class="nav-titles">
                <span class="nav-title-main">Multimodal Fake News & Deepfake Detection Platform</span>
                <span class="nav-subtitle">Department of Computer Science & Engineering • School of Computer Science & Engineering (SCOPE) • VIT-AP University</span>
            </div>
        </div>
        <div class="nav-status-group">
            <span class="nav-badge nav-badge-green"><span class="green-dot"></span> System Operational</span>
            <span class="nav-badge nav-badge-blue">Backbones: RoBERTa + EfficientNet-B0</span>
            <span class="nav-badge nav-badge-blue">Vision SOTA: YuNet ONNX</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Initializing multimodal backbones and calibrating weights..."):
        detector = load_detector()

    # ─── Sidebar Controls (Clean Enterprise Panel) ──────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Inference Configuration")
        
        fusion_strategy = st.selectbox(
            "Fusion Network Layer",
            ["cross_attention", "early", "late"],
            format_func=lambda x: {
                "cross_attention": "✨ Cross-Modal Attention (Hypothesis H1)",
                "early": "📊 Early Feature Concatenation",
                "late": "⚖️ Late Decision-Level Rule Aggregation",
            }[x],
            help="Select fusion strategy to evaluate Hypothesis H1 vs Baselines.",
        )

        st.markdown("---")
        st.markdown("### 📂 Benchmark Test Library")
        
        presets = {
            "Custom Input (Upload Your Own Media)": None,
            "1. Authentic News: NASA JWST Space Observation": {
                "headline": "James Webb Space Telescope captures deepest sharpest infrared image of distant universe.",
                "image": "demo_samples/01_Likely_Authentic_Space/jwst_deep_field.jpg",
                "type": "image"
            },
            "2. Authentic Politics: Official White House Portrait": {
                "headline": "Official photographic portrait of former President Barack Obama released by White House.",
                "image": "demo_samples/02_Likely_Authentic_Politics/official_portrait.jpg",
                "type": "image"
            },
            "3. Misleading Hoax: Venice Frozen Canals (Out-of-Context)": {
                "headline": "Historic polar vortex completely freezes canals of Venice, Italy as gondolas trapped in thick ice.",
                "image": "demo_samples/03_Potentially_Misleading_Disaster_Hoax/venice_canal.jpg",
                "type": "image"
            },
            "4. AI Face Deepfake: StyleGAN Synthetic Face (140k)": {
                "headline": "Senior technology consultant appointed to lead regional cybersecurity and infrastructure committee.",
                "image": "demo_samples/04_Potentially_Manipulated_Deepfake/stylegan_synthetic_face.jpg",
                "type": "image"
            },
            "5. Multimodal Conspiracy: Synthetic Megastructure": {
                "headline": "Secret high-tech alien lattice megastructure uncovered under ice sheet on Earth.",
                "image": "demo_samples/05_Multimodal_Fake_Conspiracy/synthetic_lattice_core.jpg",
                "type": "image"
            },
            "6. Video Deepfake: AI Manipulated Speech (MP4)": {
                "headline": "Official spokesperson press briefing on national cybersecurity infrastructure updates.",
                "video": "demo_samples/06_Video_Deepfake_Demo/deepfake_speech_sample.mp4",
                "type": "video"
            },
            "7. Authentic Video: Live Public Interview (MP4)": {
                "headline": "Former US President speaks on democratic institutions and civic engagement at leadership forum.",
                "video": "demo_samples/06_Video_Deepfake_Demo/authentic_interview_sample.mp4",
                "type": "video"
            },
            "8. Kaggle 140k Face Benchmark Sample": {
                "headline": "Prominent researcher named keynote speaker for international AI conference.",
                "image": "data/140k_faces/test/fake/00276TOPP4.jpg",
                "type": "image"
            },
            "9. Kaggle Paired Deepfake Video Sequence": {
                "headline": "Digital broadcast statement on technological innovation and autonomous systems.",
                "video": "data/kaggle_dataset/deepfake/1.mp4",
                "type": "video"
            }
        }
        
        selected_preset = st.selectbox(
            "Select Evaluation Sample:",
            list(presets.keys()),
            help="Choose a pre-validated test case with ground-truth labels."
        )

        st.markdown("---")
        st.markdown("### 👥 Project Investigators")
        st.markdown("""
        | Investigator | Reg Number | Subsystem Role |
        | :--- | :--- | :--- |
        | **Shaik Mahammad Usman** | `23BCE20183` | Video & Media Forensics |
        | **Greeshmika Ronanki** | `23BCE20074` | Vision Deepfake Branch |
        | **Meegada Nandi** | `23BCE9614` | Cross-Attention & XAI |
        | **Julakanti Surya Chaitanya Reddy** | `23BCE8832` | Text & System Integration |
        """)
        st.caption("Department of Computer Science & Engineering • VIT-AP University")

        if st.button("🔄 Flush Memory Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # ─── Resolve Selected Preset ─────────────────────────────────────────────
    default_text = ""
    preset_image = None
    preset_video_path = None
    preset_type = None

    if selected_preset != "Custom Input (Upload Your Own Media)":
        preset_info = presets[selected_preset]
        default_text = preset_info["headline"]
        preset_type = preset_info.get("type", "image")

        if preset_type == "image" and "image" in preset_info:
            img_full = os.path.join(BASE_DIR, preset_info["image"])
            if os.path.exists(img_full):
                preset_image = Image.open(img_full).convert("RGB")
        elif preset_type == "video" and "video" in preset_info:
            vid_full = os.path.join(BASE_DIR, preset_info["video"])
            if os.path.exists(vid_full):
                preset_video_path = vid_full

    # ─── Dual Input Interface (Clean Enterprise Layout) ──────────────────────
    col_text, col_media = st.columns([1.1, 0.9], gap="large")

    with col_text:
        st.markdown("""
        <div class="card-header-bar">
            <span class="card-header-title">📝 Modality 1 • Textual News Claim / Headline</span>
            <span style="font-size: 0.75rem; color: #64748B;">RoBERTa NLP Tokenizer</span>
        </div>
        """, unsafe_allow_html=True)
        input_text = st.text_area(
            "Claim Text:",
            value=default_text,
            height=165,
            placeholder="Enter the news headline, social post caption, or claim statement to analyze...",
            label_visibility="collapsed",
            help="The RoBERTa language model analyzes linguistic credibility, sensationalism, and contextual semantics."
        )
        st.caption(f"Payload: **{len(input_text)}** characters | **{len(input_text.split())}** words")

    with col_media:
        st.markdown("""
        <div class="card-header-bar">
            <span class="card-header-title">🖼️ Modalities 2, 3 & 4 • Visual / Temporal Media</span>
            <span style="font-size: 0.75rem; color: #64748B;">EfficientNet-B0 + YuNet</span>
        </div>
        """, unsafe_allow_html=True)
        
        media_mode = st.radio(
            "Media Format:",
            ["📷 Static Image Analysis", "🎥 Temporal Video Stream Analysis"],
            horizontal=True,
            index=1 if preset_type == "video" else 0,
            label_visibility="collapsed"
        )

        input_image = None
        target_video_path = None

        if "Image" in media_mode:
            uploaded_img = st.file_uploader(
                "Upload Image (PNG, JPG, JPEG, WEBP):",
                type=["png", "jpg", "jpeg", "webp"],
                key="img_upload"
            )
            if uploaded_img is not None:
                input_image = Image.open(uploaded_img).convert("RGB")
                st.image(input_image, caption=f"Source Image ({input_image.width}x{input_image.height})", use_column_width=True)
            elif preset_image is not None and preset_type == "image":
                input_image = preset_image
                st.image(input_image, caption="Preset Benchmark Image", use_column_width=True)
            else:
                st.info("💡 Upload an image file above, or select a preset from the sidebar.")
        else:
            uploaded_vid = st.file_uploader(
                "Upload Video Stream (MP4, MOV, AVI, MKV):",
                type=["mp4", "mov", "avi", "mkv"],
                key="vid_upload"
            )
            if uploaded_vid is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_vid.read())
                    target_video_path = tmp.name
                st.video(target_video_path)
            elif preset_video_path is not None and preset_type == "video":
                target_video_path = preset_video_path
                st.video(preset_video_path)
                st.caption("Preset Benchmark Video Stream Loaded")
            else:
                st.info("💡 Upload an MP4/MOV video, or select a video preset from the sidebar.")

    # ─── Analysis Action Trigger ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Execute Joint Multimodal Forensic Analysis", use_container_width=True)

    if analyze_btn:
        if not input_text.strip():
            st.error("⚠️ Textual claim is required. Please type or paste a headline to analyze.")
            return

        if input_image is None and target_video_path is None:
            st.error("⚠️ Visual media is required. Please upload an image or video, or choose a preset.")
            return

        with st.spinner("Executing neural reasoning across spatial CNN, 2D-FFT frequency spectrum, and cross-attention..."):
            is_video_mode = target_video_path is not None and input_image is None

            if is_video_mode:
                video_analyzer = VideoAnalyzer(detector=detector, num_frames=8)
                result = video_analyzer.analyze(
                    target_video_path, text=input_text, fusion_strategy=fusion_strategy
                )
            else:
                result = detector.predict(
                    text=input_text,
                    image=input_image,
                    fusion_strategy=fusion_strategy,
                )

        # ─── 1. Executive Verdict Banner ─────────────────────────────────────
        verdict = result["verdict"]
        confidence = result["confidence"]

        theme_map = {
            "Likely Authentic": ("vb-auth", "#10B981", "VERIFIED AUTHENTIC CONTENT"),
            "Potentially Misleading": ("vb-mislead", "#F59E0B", "OUT-OF-CONTEXT / CONTEXT MISMATCH"),
            "Potentially Manipulated": ("vb-manip", "#EF4444", "SYNTHETIC MANIPULATION / DEEPFAKE"),
            "Multimodal Fake Content": ("vb-multi", "#A855F7", "FABRICATED MULTIMODAL DISINFORMATION"),
        }
        theme_class, theme_color, theme_badge = theme_map.get(
            verdict, ("vb-auth", "#3B82F6", "ANALYSIS COMPLETE")
        )

        st.markdown(f"""
        <div class="verdict-banner {theme_class}">
            <div class="verdict-left">
                <span class="verdict-tag" style="color: {theme_color};">{theme_badge}</span>
                <h2 class="verdict-name">{verdict}</h2>
                <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
                    Architecture: <b>{result['fusion_strategy'].replace('_', ' ').title()}</b> &bull; 
                    Modality: <b>{'Temporal Video Stream' if is_video_mode else 'Static Image'}</b>
                </div>
            </div>
            <div class="verdict-metrics-row">
                <div class="v-metric-item">
                    <span class="v-metric-label">Classification Confidence</span>
                    <span class="v-metric-value" style="color: {theme_color};">{confidence:.1%}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── 2. Modality 4-Pillars Diagnostic Grid ───────────────────────────
        st.markdown("""
        <div class="card-header-bar">
            <span class="card-header-title">📊 Multi-Pillar Diagnostic Telemetry</span>
            <span style="font-size: 0.75rem; color: #64748B;">4-Dimensional Forensic Breakdown</span>
        </div>
        """, unsafe_allow_html=True)
        p_cols = st.columns(4)

        mod_scores = result["modality_scores"]
        pillar_data = MultimodalExplainer.format_modality_breakdown(mod_scores)

        pillar_icons = {
            "Text Credibility": "📝",
            "Vision Deepfake": "👁️",
            "Forensic FFT Spectrum": "🔬",
            "Cross-Modal Consistency": "🔗",
        }

        for i, (name, p_info) in enumerate(pillar_data.items()):
            with p_cols[i]:
                st.markdown(f"""
                <div class="pillar-box">
                    <div class="pillar-box-label">{pillar_icons.get(name, '📊')} {name}</div>
                    <div class="pillar-box-num" style="color: {p_info['color']};">{p_info['score']:.1%}</div>
                    <span class="pillar-box-status" style="background: {p_info['color']}18; color: {p_info['color']}; border: 1px solid {p_info['color']}35;">
                        {p_info['label']}
                    </span>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 6px;">Risk Index: <b>{p_info['risk']}</b></div>
                </div>
                """, unsafe_allow_html=True)

        # ─── 3. Video Temporal Forensics (If Video Mode) ─────────────────────
        if is_video_mode and "video" in result:
            v_meta = result["video"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card-header-bar">
                <span class="card-header-title">⏱️ Temporal Video Forensics • Keyframe & Jitter Diagnostics</span>
                <span style="font-size: 0.75rem; color: #64748B;">Multi-Frame Uniform Sampling</span>
            </div>
            """, unsafe_allow_html=True)

            v_col1, v_col2 = st.columns([1, 1.2], gap="large")
            
            with v_col1:
                if "key_frame" in result and result["key_frame"] is not None:
                    st.image(
                        result["key_frame"],
                        caption=f"Argmax Suspicious Keyframe (Timestamp t={v_meta['key_frame_time_sec']:.2f}s)",
                        use_column_width=True
                    )
                st.markdown(f"""
                <div style="display: flex; gap: 8px; justify-content: center; margin-top: 8px;">
                    <span class="nav-badge nav-badge-blue">Sampled: {v_meta['total_frames_sampled']} frames</span>
                    <span class="nav-badge" style="{'background:rgba(239,68,68,0.15); color:#F87171; border:1px solid rgba(239,68,68,0.3);' if v_meta['is_flagged'] else 'background:rgba(16,185,129,0.15); color:#34D399; border:1px solid rgba(16,185,129,0.3);'}">
                        {'🚨 Sequence Flagged' if v_meta['is_flagged'] else '✅ Clean Temporal Sequence'}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with v_col2:
                # Video Metrics
                vm1, vm2, vm3 = st.columns(3)
                with vm1:
                    st.metric("Aggregated Video Risk", f"{v_meta['video_risk']:.1%}", help="Top-3 frame risk average")
                with vm2:
                    st.metric("Temporal Jitter (σ_Δ)", f"{v_meta['temporal_jitter']:.4f}", help="Standard deviation of inter-frame changes")
                with vm3:
                    st.metric("Flagged Ratio", f"{v_meta['flagged_frame_ratio']:.0%}", help="Frames with risk >= threshold")

                # Line chart of frame risks
                st.markdown("##### Per-Frame Deepfake Risk Curve Across Sequence")
                frame_risks = [p["risk"] for p in v_meta["per_frame"]]
                df_chart = pd.DataFrame({
                    "Frame Risk Score": frame_risks,
                    "Alert Threshold (0.55)": [0.55] * len(frame_risks)
                })
                st.line_chart(df_chart, height=190)

        # ─── 4. Detailed Explainability Tabs ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card-header-bar">
            <span class="card-header-title">🔬 Explainable AI (XAI) • Multimodal Evidence Synthesis</span>
            <span style="font-size: 0.75rem; color: #64748B;">Auditability & Decision Provenance</span>
        </div>
        """, unsafe_allow_html=True)

        tab_reasoning, tab_spectrum, tab_text_attn, tab_fusion = st.tabs([
            "🧠 Forensic Decision Synthesis",
            "🔬 2D-FFT Power Spectrum Heatmap",
            "✍️ Linguistic Token Salience",
            "📊 Softmax Probability Distribution",
        ])

        with tab_reasoning:
            exp_text = result.get("explanation", "")
            if isinstance(exp_text, dict):
                exp_text = exp_text.get("summary", str(exp_text))
            
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.75); border-left: 3px solid #3B82F6; border-radius: 10px; padding: 1.4rem; color: #CBD5E1; font-size: 0.95rem; line-height: 1.8; border: 1px solid rgba(255,255,255,0.06); border-left-color: #3B82F6;">
                <div style="font-weight: 700; color: #60A5FA; margin-bottom: 0.5rem; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.8px;">
                    Executive Decision Summary
                </div>
                {exp_text}
            </div>
            """, unsafe_allow_html=True)

        with tab_spectrum:
            col_s1, col_s2 = st.columns([1, 1.2], gap="large")
            with col_s1:
                if "spectrum_image" in result and result["spectrum_image"] is not None:
                    st.image(
                        result["spectrum_image"],
                        caption="2D-FFT Log-Magnitude Azimuthal Power Spectrum",
                        use_column_width=True
                    )
            with col_s2:
                st.markdown("#### High-Frequency Lattice Diagnostics")
                f_details = result["branch_details"]["forensic"]
                st.markdown(f"""
                * **High-Frequency Energy:** `{f_details.get('high_freq_energy', 0.0):.4f}`
                * **Mid-Frequency Energy:** `{f_details.get('mid_freq_energy', 0.0):.4f}`
                * **Synthetic Anomaly Score:** `{mod_scores.get('forensic_synthetic_score', 0.0):.1%}`
                """)
                st.info(
                    "Natural optical sensors exhibit smooth, continuous 1/f radial power decay. In contrast, generative architectures (GAN upsampling, StyleGAN, Diffusion) produce distinct periodic spikes and grid anomalies across the frequency perimeter."
                )

        with tab_text_attn:
            st.markdown("#### Self-Attention Token Attribution")
            text_details = result["branch_details"]["text"]
            word_html = MultimodalExplainer.generate_word_html(text_details.get("word_attributions", []))
            if word_html:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.75); padding: 1.2rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.07); line-height: 2.2;">
                    {word_html}
                </div>
                """, unsafe_allow_html=True)
                st.caption("Highlighted tokens indicate words with the highest self-attention weights driving RoBERTa veracity prediction.")
            else:
                st.write(text_details.get("text", input_text))

        with tab_fusion:
            st.markdown("#### Multimodal Softmax Probability Distribution")
            for cat, prob in result["probabilities"].items():
                col_c1, col_c2 = st.columns([1.2, 4])
                with col_c1:
                    st.markdown(f"**{cat}**")
                with col_c2:
                    st.progress(float(prob), text=f"{prob:.1%}")

    # ─── Academic Capstone Footer ────────────────────────────────────────────
    st.markdown("""
    <div class="pro-footer">
        <b>Final Year Capstone Project (2026)</b> &bull; Multimodal Fake News and Deepfake Detection System Using Deep Learning and Explainable AI<br>
        Department of Computer Science & Engineering &bull; School of Computer Science & Engineering (SCOPE) &bull; <b>VIT-AP University</b><br>
        Investigators: <b>Shaik Mahammad Usman</b> (23BCE20183) &bull; <b>Greeshmika Ronanki</b> (23BCE20074) &bull; <b>Meegada Nandi</b> (23BCE9614) &bull; <b>Julakanti Surya Chaitanya Reddy</b> (23BCE8832)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
