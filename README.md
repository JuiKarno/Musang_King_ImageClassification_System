# 🍈 MusangKing Classification System

A professional Flask-based web application for durian (Musang King vs D24) classification using computer vision techniques.

![Dashboard Screenshot](static/uploads/screenshot.png)

## 🎯 Features

- **Image Upload**: Drag-and-drop or click to upload durian images
- **Interactive Parameters**: Adjustable Gamma correction (0.5-2.0) and K-Means clusters (2-6)
- **Real-time Visualization**: Side-by-side Original Image & Processed Mask display
- **Feature Extraction**: Compactness, Smoothness, and Mean Hue analysis
- **Classification**: Automatic detection of Musang King (Mature) vs D24 (Immature)

## 📁 Project Structure

```
/MusangKing_System
├── app.py                    # Flask backend with Phase 1-4 processing
├── requirements.txt          # Dependencies
├── static/
│   ├── css/style.css        # Modern dark theme styling
│   ├── js/main.js           # Frontend interactivity
│   └── uploads/             # Uploaded & processed images
└── templates/
    └── index.html           # 3-panel dashboard layout
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Bukhhh/MusangKing_System.git
cd MusangKing_System
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
Navigate to **http://127.0.0.1:5000**

## 🔬 Image Processing Pipeline

| Phase | Description |
|-------|-------------|
| **Phase 1** | Preprocessing - Gamma Correction & LAB Color Space |
| **Phase 2** | Segmentation - K-Means Clustering |
| **Phase 3** | Refinement - Morphological Operations |
| **Phase 4** | Feature Extraction - Compactness, Smoothness, Mean Hue |

## 📊 Classification Output

The system analyzes durian images and provides:
- **Compactness**: Shape circularity measure (0-1)
- **Smoothness**: Surface smoothness index (0-1)
- **Mean Hue**: Average color hue value (degrees)
- **Classification**: "Musang King (Mature)" or "D24 (Immature)"

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **Image Processing**: OpenCV, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with glassmorphism effects

## 👥 Team

CSC566 Image Processing Project - Group Musang King

## 📝 License

This project is for educational purposes.

---

*Built for Week 14 Presentation & Final Report*
