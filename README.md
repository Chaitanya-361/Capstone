# Multimodal Fake News & Deepfake Detection System

> **Capstone Project — VIT-AP University**  
> Multimodal Fake News and Deepfake Detection Using Deep Learning and Explainable AI  
> Aligned with PRD System Architecture (Hypotheses H1 & H2)

---

## 🚀 Overview

This end-to-end multimodal intelligence system detects fake news, synthetic/deepfake imagery, and out-of-context cross-modal mismatches. By leveraging pre-trained state-of-the-art backbones, it delivers real-time inference on CPU with zero compute overhead, eliminating the need for expensive GPU training from scratch.

### 4-Way Output Taxonomy (PRD Section 4.2)
1. **Likely Authentic** — Real image + factual, verified text context.
2. **Potentially Misleading** — Genuine photo used out-of-context, or sensational text claim.
3. **Potentially Manipulated** — Deepfake, FaceForensics++ manipulation, or GAN/Diffusion artifact detected.
4. **Multimodal Fake Content** — Manipulated image coupled with fabricated or deceptive text.

---

## 🏛️ System Architecture

```
                    ┌───────────────────────────────┐
                    │      Multimodal Input         │
                    │   (Text Claim + Image File)   │
                    └───────────────┬───────────────┘
                                    │
         ┌──────────────────────────┼─────────────────────────┐
         ▼                          ▼                         ▼
┌─────────────────┐        ┌─────────────────┐       ┌─────────────────┐
│   Text Branch   │        │  Vision Branch  │       │ Forensic Branch │
│  (RoBERTa-base) │        │ (EfficientNet-  │       │ (2D-FFT Power   │
│  768-dim CLS    │        │  B0 on FF++c23) │       │  Spectrum Bins) │
│  Self-Attention │        │  768-dim Visual │       │  128-dim Radial │
└────────┬────────┘        └────────┬────────┘       └────────┬────────┘
         │                          │                         │
         ├──────────────────────────┤                         │
         │                          │                         │
         ▼                          ▼                         │
┌────────────────────────────────────────────┐                │
│       Cross-Modal Consistency Branch       │                │
│   (Cosine Similarity in Joint 512-d Space) │                │
└─────────────────────┬──────────────────────┘                │
                      │                                       │
                      ▼                                       ▼
         ┌────────────────────────────────────────────────────────┐
         │             Multi-Head Cross-Attention Fusion          │
         │   • Text-to-Vision & Vision-to-Text Cross-Attention    │
         │   • Baselines: Early Fusion & Late Decision Fusion     │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │             Explainable AI (XAI) Output                │
         │   • 4-Class Softmax Probability Distribution           │
         │   • Word-Level Attention Highlight Tokens              │
         │   • 2D Frequency Power Spectrum Heatmap                │
         │   • Natural Language Grounded Verdict Summary          │
         └────────────────────────────────────────────────────────┘
```

---

## 📦 Branches & Models Integrated

| Branch | Backbone Model | Pre-training Source | Dimension | Output / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Text Branch** | RoBERTa-base | Hugging Face Transformers | 768-d | Feature vector, claim suspicion score, word attributions |
| **Vision Branch** | EfficientNet-B0 | Fine-tuned on **FaceForensics++ (c23)** (`Xicor9/efficientnet-b0-ffpp-c23`) | 768-d | Deepfake probability, facial/visual forgery features |
| **Forensic Branch** | 2D-FFT Azimuthal Integrator | Frequency Domain Signal Processing (OpenCV/NumPy) | 128-d | High-frequency lattice & GAN upsampling score, 2D power spectrum heatmap |
| **Consistency** | Cross-Modal Metric Projection | Multi-layer Perceptron (768 ➔ 512-d) | 1-d | Text-image semantic alignment score (catches out-of-context reuse) |
| **Fusion Layer** | Bidirectional Multi-Head Cross-Attention | Scaled Dot-Product Cross-Attention + GELU Classifier | 4 classes | Final calibrated verdict, class probabilities, cross-modal attention maps |

---

## 🔬 Fusion Hypotheses Tested (PRD Section 3.2)

The system includes 3 selectable fusion strategies:
- **`cross_attention` (Primary, Hypothesis H1):** Multi-head cross-attention where text attends to visual tokens and visual tokens attend to text tokens. Outperforms simple concatenation.
- **`early` (Baseline 1):** Direct concatenation of feature embeddings `[Text; Vision; Forensic; Consistency]` fed into a dense classifier.
- **`late` (Baseline 2):** Rule-based decision-level aggregation of individual modality confidence scores.

---

## 🛠️ Installation & Verification

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Integration Test Suite
To verify all 5 pipeline components, tensor dimensions, and fusion heads:
```bash
python tests/test_system.py
```
Expected output:
```text
============================================================
  Multimodal Fake News Detection System — Verification
============================================================
[1/5] Testing Text Branch (RoBERTa)... [OK] PASS! Features (768,)
[2/5] Testing Vision Branch (EfficientNet-B0 FF++)... [OK] PASS! Features (768,)
[3/5] Testing Forensic Branch (2D-FFT)... [OK] PASS! Features (128,)
[4/5] Testing Cross-Attention Fusion & Consistency... [OK] PASS! Logits (1, 4)
[5/5] Testing End-to-End Detector... [OK] PASS!
============================================================
  ALL 5/5 INTEGRATION TESTS PASSED PERFECTLY!
============================================================
```

---

## 🖥️ Interactive Web Dashboard

Launch the Streamlit web dashboard:
```bash
streamlit run frontend/app.py
```

### Dashboard Features:
- **Dual Input Interface:** Enter any news headline / claim and upload an image (PNG, JPG, WEBP) or pick from built-in sample presets.
- **Fusion Strategy Selector:** Toggle between **Cross-Attention**, **Early Fusion**, and **Late Fusion** live to compare predictions.
- **Verdict Card:** Prominent color-coded classification with confidence percentage.
- **4 Modality Score Pillars:** Live gauges for Text Suspicion, Deepfake Probability, Frequency Synthetic Score, and Cross-Modal Consistency.
- **Explainable AI (XAI) Panel:**
  - Token-level text attribution highlighting the most influential words.
  - Interactive 2D-FFT frequency spectrum heatmap showing synthetic periodic lattice artifacts.
  - Natural language synthesis explaining *why* the verdict was reached.

---

## 👥 Project Team (VIT-AP University)

| Member | ID | Ownership |
| :--- | :--- | :--- |
| **Shaik Mahammad Usman** | 23BCE20183 | Text Branch & NLP Feature Extraction |
| **Greeshmika Ronanki** | 23BCE20074 | Vision Branch & FaceForensics++ Deepfake Detection |
| **Meegada Nandi** | 23BCE9614 | Cross-Attention Fusion & Explainable AI (XAI) |
| **Julakanti Surya Chaitanya Reddy** | 23BCE8832 | Forensic Frequency Analysis & System Integration |
