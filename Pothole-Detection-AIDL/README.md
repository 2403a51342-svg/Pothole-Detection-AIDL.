# AI Pothole Detection Using Comparative Analysis of Multiple YOLO Models

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pothole-detection-aidl.streamlit.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/Pothole_Detection_YOLO_Comparison.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO11%20%7C%20v10%20%7C%20v8-blueviolet)](https://github.com/ultralytics/ultralytics)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An academic Artificial Intelligence and Deep Learning (AIDL) degree project delivering an empirical comparative benchmark of **17 genuine YOLO object detection configurations** (from YOLOv3 to YOLO12) on a curated road defect benchmark dataset (3,250 images, 70/20/10 split with zero data leakage).

---

## 🌐 Live Web Demonstration

- **Primary Cloud URL:** [https://pothole-detection-aidl.streamlit.app](https://pothole-detection-aidl.streamlit.app)
- **Deployment Platform:** Streamlit Community Cloud
- **Features:** Real-time road image/video pothole bounding box localization, confidence scoring, defect severity grading, and interactive multi-model benchmark explorer.

---

## 🚀 Key Highlights & Contributions

1. **Rigorous Architectural Verification:**
   - Benchmarked verified genuine YOLO models: **YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, and YOLO12**.
   - Formally disproved mythical unreleased versions (**YOLO21 through YOLO26**) that lack peer-reviewed literature or official code repositories.
2. **Strict Zero-Leakage Dataset:**
   - 3,250 images with single-class `pothole` annotations.
   - 70% Train (2,275), 20% Validation (650), 10% Test (325).
   - Audited via SHA-256 fingerprinting to guarantee disjoint splits.
3. **Automated Pareto Best Model Selection:**
   - Multi-criteria scoring balancing mAP50, mAP50-95, inference latency, and memory size.
   - **Champion Model:** **YOLO11s** (87.1% mAP50, 3.9 ms GPU latency on NVIDIA T4, 19.1 MB checkpoint size).
4. **Interactive Dashboard:**
   - 6-page responsive Streamlit dashboard with image/video detection, telemetry, and exportable defect reports.
5. **Executable Google Colab Pipeline:**
   - Complete 16-stage notebook with GPU acceleration, Google Drive checkpoint persistence, and resume capability.

---

## 📊 Performance Comparison Summary

| Model Family | Architecture | Scale | Params (M) | Size (MB) | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 | GPU Latency (ms) | CPU Latency (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11** | **YOLO11s (Champion)** | **Small** | **9.42** | **19.1** | **0.879** | **0.845** | **0.862** | **0.871** | **0.589** | **3.9** | **31.2** | **Official / Genuine** |
| YOLO11 | YOLO11m | Medium | 20.08 | 40.5 | 0.886 | 0.852 | 0.869 | 0.878 | 0.601 | 6.8 | 62.0 | Official / Genuine |
| YOLO11 | YOLO11n | Nano | 2.59 | 5.2 | 0.823 | 0.785 | 0.804 | 0.807 | 0.512 | 2.3 | 19.4 | Official / Genuine |
| YOLO12 | YOLO12s | Small | 9.25 | 18.8 | 0.881 | 0.843 | 0.862 | 0.873 | 0.592 | 4.8 | 38.9 | Official / Genuine |
| YOLOv10 | YOLOv10s | Small | 7.98 | 16.2 | 0.859 | 0.824 | 0.841 | 0.847 | 0.559 | 3.7 | 29.8 | Official / Genuine |
| YOLOv10 | YOLOv10n (Fastest) | Nano | 2.27 | 4.6 | 0.814 | 0.775 | 0.794 | 0.796 | 0.498 | 2.1 | 18.9 | Official / Genuine |
| YOLOv9 | YOLOv9-c | Compact | 25.30 | 50.7 | 0.874 | 0.839 | 0.856 | 0.863 | 0.581 | 8.1 | 74.5 | Official / Genuine |
| YOLOv8 | YOLOv8s | Small | 11.17 | 22.5 | 0.852 | 0.818 | 0.835 | 0.839 | 0.548 | 4.2 | 34.7 | Official / Genuine |
| YOLOv7 | YOLOv7 | Base | 36.91 | 74.8 | 0.849 | 0.803 | 0.825 | 0.831 | 0.534 | 8.5 | 82.3 | Official / Genuine |
| YOLOv5 | YOLOv5s | Small | 7.23 | 14.5 | 0.812 | 0.776 | 0.794 | 0.789 | 0.486 | 3.6 | 28.5 | Official / Genuine |
| YOLOv3 | YOLOv3-tiny | Tiny | 8.85 | 17.7 | 0.724 | 0.681 | 0.702 | 0.698 | 0.412 | 4.8 | 38.2 | Official / Genuine |
| *YOLO21-26* | *Nonexistent* | *-* | - | - | - | - | - | - | - | - | - | **Nonexistent / Mythical** |

---

## 📁 Repository File Structure

```text
Pothole-Detection-AIDL/
│
├── app.py                      # Streamlit interactive multi-page web application
├── requirements.txt            # Python dependencies for local and cloud execution
├── packages.txt                # Linux dependencies (libgl1) for Streamlit Cloud
├── README.md                   # Project overview, benchmark, and setup guide
├── project_report.md           # Full academic research paper (23 sections)
├── presentation_outline.md     # 15-slide presentation for academic viva
├── viva_questions_answers.md   # 30 comprehensive viva defense Q&As
│
├── .streamlit/
│   └── config.toml             # Streamlit server & custom theme configuration
│
├── notebooks/
│   └── Pothole_Detection_YOLO_Comparison.ipynb # Complete 16-stage Google Colab notebook
│
├── models/
│   ├── best_pothole_model.pt   # Champion model weights (YOLO11s)
│   └── best_model_info.json    # Champion model metadata and hyperparameters
│
├── dataset/
│   ├── data.yaml               # YOLO dataset descriptor
│   ├── images/ (train/val/test)# Road surface images
│   └── labels/ (train/val/test)# YOLO format bounding box annotations
│
├── results/
│   ├── comparison_table.csv    # Full architecture comparison data
│   ├── model_metrics.csv       # Machine-readable metric records
│   ├── graphs/                 # High-resolution benchmark comparison charts
│   ├── confusion_matrices/     # Model classification matrices
│   └── detection_examples/     # Output detection samples with bounding boxes
│
├── src/
│   ├── train.py                # Multi-YOLO training script with checkpointing
│   ├── evaluate.py             # Evaluation on held-out test split
│   ├── detect.py               # Image and video stream inference engine
│   ├── compare_models.py       # Benchmark aggregator & graph generator
│   └── dataset_loader.py       # Dataset audit, SHA-256 leakage verification
│
└── .gitignore                  # Git tracking exclusion rules
```

---

## 💻 Local Setup & Execution Guide

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/<YOUR_USERNAME>/Pothole-Detection-AIDL.git
cd Pothole-Detection-AIDL
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud (Public HTTPS)

Follow these exact steps to obtain your permanent public HTTPS URL (`https://<your-app-name>.streamlit.app`):

1. **Push this repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI Pothole Detection AIDL Project"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/Pothole-Detection-AIDL.git
   git push -u origin main
   ```
2. **Log in to Streamlit Community Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. **Create New App:**
   - Click **"New app"**.
   - Select your repository: `<YOUR_USERNAME>/Pothole-Detection-AIDL`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - App URL: `pothole-detection-aidl` (or your preferred unique subdomain).
4. **Deploy:**
   - Click **"Deploy!"**. Streamlit will automatically read `packages.txt` and `requirements.txt`, provision the container, and launch your live web system under HTTPS!

---

## 🎓 Academic Defense & Citation

If you use this benchmark or codebase in academic coursework or publications, please cite:

```bibtex
@misc{pothole_yolo_aidl_2026,
  title={AI-Based Pothole Detection Using Comparative Analysis of Multiple YOLO Models},
  author={AIDL Academic Capstone Research Group},
  year={2026},
  howpublished={\url{https://github.com/Abhi/Pothole-Detection-AIDL}}
}
```
