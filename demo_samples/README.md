# 🧪 Multimodal Fake News & Deepfake Detection — Demo Cheat Sheet

This folder contains pre-configured test cases representing each category in the PRD taxonomy.
Use these to manually test and demonstrate the system in the Streamlit app.

---

### Quick Copy-Paste Testing Guide

| # | Case Folder | Expected Verdict | Headline to Copy-Paste | Image File to Upload | Key Signals |
|---|---|---|---|---|---|
| **1** | `01_Likely_Authentic_Space` | **Likely Authentic** | `NASA's James Webb Space Telescope captures deepest infrared image of early galaxies in cluster SMACS 0723.` | `jwst_deep_field.jpg` | Text Credibility ~95%, Deepfake 0.0%, Natural Sensor Noise |
| **2** | `02_Likely_Authentic_Politics` | **Likely Authentic** | `Former US President speaks on democratic institutions and civic engagement at international leadership forum.` | `official_portrait.jpg` | Text Credibility ~92%, Deepfake 0.0%, Natural Skin Texture |
| **3** | `03_Potentially_Misleading_Disaster_Hoax` | **Potentially Misleading** | `BREAKING: Catastrophic 8.5 magnitude tsunami completely inundates coastal city center as terrified residents evacuate on boats!` | `venice_canal.jpg` | Real calm photo + sensational false disaster claim (Out-of-context) |
| **4** | `04_Potentially_Manipulated_Deepfake` | **Potentially Manipulated** | `Senior technology consultant appointed to lead regional cybersecurity and infrastructure committee.` | `stylegan_synthetic_face.jpg` | Credible neutral text + FaceForensics++ 100% Deepfake flag |
| **5** | `05_Multimodal_Fake_Conspiracy` | **Multimodal Fake Content** | `SHOCKING EXPOSÉ: Classified extraterrestrial energy core retrieved from secret underground bunker by military command!` | `synthetic_lattice_core.jpg` | Sensational conspiracy text + High-frequency GAN upsampling grid |

---

### How to Run a Manual Demo:
1. Open the running dashboard at `http://localhost:8501`.
2. Open any subfolder inside `demo_samples/`.
3. Open `headline.txt` and copy the text into the **Text Modality** box in the browser.
4. Drag and drop the corresponding image file into the **Visual Modality** upload box.
5. Click **"🚀 Run Multimodal Joint Analysis"**.
