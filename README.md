# 👑 Musang King Image Classification System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Large%20Files-purple?logo=git)

> An intelligent image classification system that identifies **Durian variety** and **ripeness level** using a hybrid computer vision pipeline — built for **CSC566 Advanced Image Processing** at UiTM (Oct 2025 – Mar 2026).

---

## 🧠 How It Works

Instead of deep learning, this system uses a lightweight **8-Step Hybrid Pipeline** to extract interpretable features from durian images:

| Step | Process | Purpose |
|------|---------|---------|
| 1 | Gamma Correction | Normalize lighting |
| 2 | LAB Color Space | Enhance color separation |
| 3 | K-Means Clustering (K=3) | Segment husk vs. flesh |
| 4 | Morphological Cleanup | Refine segmentation masks |
| 5 | Contour Detection | Locate the durian object |
| 6 | Feature Extraction | Compute geometric features (compactness, aspect ratio) |
| 7 | Color Analysis | RGB/HSV histogram of husk |
| 8 | Binary Masking | Isolate region of interest (ROI) |

An **XGBoost classifier** is then trained on these extracted features for fast and accurate prediction.

---

## 🎯 Classification Targets

- **Variety**: Musang King (D197), Black Thorn (D200), Udang Merah (D175)
- **Ripeness**: Mature, Immature, Defective

---

## 📊 Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Musang King | 0.95 | 0.91 | 0.93 |
| Black Thorn | 0.96 | 0.93 | 0.94 |
| Udang Merah | 0.88 | 0.87 | 0.87 |
| **Overall** | **88%+** | — | — |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10, Flask
- **Computer Vision**: OpenCV
- **Machine Learning**: XGBoost, scikit-learn
- **Model Storage**: Git LFS (`.pkl` files)

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/JuiKarno/Musang_King_ImageClassification_System.git
cd Musang_King_ImageClassification_System

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open your browser at: `http://127.0.0.1:5000`

---

## 📂 Project Structure

```
├── app.py                    # Main Flask application
├── train_aligned_model.py    # Model training script
├── debug_model.py            # Model health checker
├── requirements.txt          # Python dependencies
│
├── models_v4_noscale/        # Trained XGBoost models (via Git LFS)
│   ├── variety_model.pkl
│   └── ripeness_model.pkl
│
├── static/                   # CSS, JS, images
└── templates/                # HTML pages
    ├── index.html            # Dashboard
    ├── processing_stages.html # Pipeline visualization
    ├── datasets.html         # Model performance info
    ├── documentation.html    # Technical docs
    └── about.html            # Team info
```

---

## 👥 Team — Group MusangKing

| Name | Role |
|------|------|
| Yazid Zaqwan Hakim | Team Member |
| Muhamad Zulkarnain | Team Member |
| Amirul Fariz | Team Member |
| Mohamad Bukhari | Team Member |

**Supervisor**: Dr. Zaaba Ahmad (FSKM, UiTM)

---

## 📜 License

This project is developed for educational purposes under **Universiti Teknologi MARA (UiTM)**.
