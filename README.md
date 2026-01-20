# 👑 MusangKing Hybrid Classification System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0-red?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

An integrated image processing system for the classification of Durian Ripeness and Variety using a **Hybrid Pipeline** (K-Means Clustering + Morphological Feature Extraction + XGBoost Classification).

Developed for **CSC566 - Advanced Image Processing** (Oct 2025 – Mar 2026).

---

## 🌟 Key Features

### 1. Hybrid Processing Pipeline
Instead of relying solely on Deep Learning, this system uses an **8-Step Image Processing Pipeline** to extract interpretable features:
1.  **Gamma Correction**: Normalizes lighting conditions.
2.  **LAB Color Space**: Enhances color separation (Flesh vs. Husk).
3.  **K-Means Clustering**: Unsupervised segmentation (K=3).
4.  **Morphological Cleanup**: Closing operations to refine masks.
5.  **Contour Detection**: Localizes the primary durian object.
6.  **Feature Extraction**: Calculates Geometric (Compactness, Aspect Ratio) features.
7.  **Color Analysis**: Computes RGB/HSV Histograms of the husk.
8.  **Binary Masking**: Isolates the Region of Interest (ROI).

### 2. Multi-Class Classification
*   **Variety**: Musang King (D197), Black Thorn (D200), Udang Merah (D175).
*   **Ripeness**: Mature, Immature, Defective.

### 3. Detailed Visualization
*   **Dedicated "Processing Stages" Page**: Visualizes every step of the algorithm.
*   **Interactive Modal**: See exactly how the computer "sees" the fruit.
*   **Explainable AI**: Real-time display of calculated feature metrics.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Git

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/MusangKing_GUI.git
    cd MusangKing_GUI
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```
5.  **Open in Browser**
    Visit `http://127.0.0.1:5000`

---

## 📂 Project Structure

```
MusangKing_GUI/
├── app.py                  # Main Flask Application
├── debug_model.py          # Script to check model health
├── train_aligned_model.py  # Script used to train the models
├── requirements.txt        # Python dependencies
│
├── models_v4_noscale/      # Trained XGBoost Models & Encoders
│   ├── variety_model.pkl
│   ├── ripeness_model.pkl
│   └── ...
│
├── static/
│   ├── css/                # Stylesheets (Dark Mode support)
│   ├── js/                 # Frontend Logic
│   └── images/             # Assets (Logos, Team Photos)
│
└── templates/
    ├── index.html              # Dashboard
    ├── processing_stages.html  # Pipeline Visualization
    ├── datasets.html           # Model Performance Info
    ├── documentation.html      # Technical Docs
    └── about.html              # Team Info
```

---

## 🧠 Model Performance

| Variety | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Musang King** | 0.95 | 0.91 | 0.93 |
| **Black Thorn** | 0.96 | 0.93 | 0.94 |
| **Udang Merah** | 0.88 | 0.87 | 0.87 |
| **Overall Accuracy** | **88%+** | (Hybrid Pipeline) | |

---

## 👥 The Team

**Group MusangKing**
*   **Yazid Zaqwan Hakim**
*   **Muhamad Zulkarnain**
*   **Amirul Fareez**
*   **Mohamad Bukhari**

**Supervisor**
*   **Dr. Zaaba Ahmad** (FSKM, UiTM)

---

## 📜 License
This project is for educational purposes under UiTM.
