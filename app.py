from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# PLACEHOLDER FUNCTIONS - Replace with actual Phase 1-4 logic
# ============================================================

def phase1_preprocessing(image, gamma=1.0):
    """
    Phase 1: Preprocessing
    - Convert to LAB color space
    - Apply Gamma correction
    
    TODO: Replace with actual logic from Colab notebook
    """
    # Placeholder: Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(image, table)
    
    # Convert to LAB
    lab_image = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    return lab_image, corrected

def phase2_segmentation(image, k=3):
    """
    Phase 2: K-Means Segmentation
    - Isolate durian flesh from background
    
    TODO: Replace with actual K-Means logic from Colab notebook
    """
    # Placeholder: Simple K-Means clustering
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, 
                                     cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(image.shape)
    
    return segmented_image, labels.reshape(image.shape[:2])

def phase3_refinement(labels, image_shape):
    """
    Phase 3: Morphological Refinement
    - Apply opening/closing operations
    - Clean up the mask
    
    TODO: Replace with actual refinement logic from Colab notebook
    """
    # Placeholder: Create binary mask and apply morphological operations
    mask = np.uint8((labels == labels.max()) * 255)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def phase4_feature_extraction(original_image, mask):
    """
    Phase 4: Feature Extraction
    - Calculate Compactness
    - Calculate Smoothness
    - Calculate Mean Hue
    
    TODO: Replace with actual feature extraction logic from Colab notebook
    """
    # Find contours for compactness calculation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {"compactness": 0, "smoothness": 0, "mean_hue": 0}
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Compactness: 4 * pi * Area / Perimeter^2 (perfect circle = 1)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Smoothness: Using contour approximation ratio
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    smoothness = len(approx) / len(largest_contour) if len(largest_contour) > 0 else 0
    smoothness = 1 - min(smoothness, 1)  # Invert so higher = smoother
    
    # Mean Hue: Convert to HSV and extract hue channel
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    masked_hue = cv2.bitwise_and(hsv[:, :, 0], hsv[:, :, 0], mask=mask)
    mean_hue = np.mean(masked_hue[mask > 0]) if np.sum(mask > 0) > 0 else 0
    
    return {
        "compactness": round(compactness, 4),
        "smoothness": round(smoothness, 4),
        "mean_hue": round(mean_hue, 2)
    }

def classify_durian(features):
    """
    Classification Logic
    - Determine if Musang King (Mature) or D24 (Immature)
    
    TODO: Replace with actual classification thresholds from your research
    """
    # Placeholder thresholds - REPLACE WITH ACTUAL VALUES
    # These are example values and need to be calibrated with real data
    
    compactness = features['compactness']
    smoothness = features['smoothness']
    mean_hue = features['mean_hue']
    
    # Example classification logic (PLACEHOLDER)
    # Musang King typically has: higher compactness, smoother surface, specific hue range
    score = 0
    
    if compactness > 0.5:
        score += 1
    if smoothness > 0.3:
        score += 1
    if 20 <= mean_hue <= 40:  # Yellow-ish hue range
        score += 1
    
    if score >= 2:
        return "Musang King (Mature)", "musang-king"
    else:
        return "D24 (Immature)", "d24"

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique filename
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': f'/static/uploads/{filename}'
    })

@app.route('/process', methods=['POST'])
def process_image():
    """Process the uploaded image through Phase 1-4"""
    data = request.get_json()
    filename = data.get('filename')
    gamma = float(data.get('gamma', 1.0))
    k_value = int(data.get('k_value', 3))
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found'}), 404
    
    # Read the original image
    original = cv2.imread(filepath)
    if original is None:
        return jsonify({'error': 'Could not read image'}), 400
    
    # Phase 1: Preprocessing
    lab_image, gamma_corrected = phase1_preprocessing(original, gamma)
    
    # Phase 2: Segmentation
    segmented, labels = phase2_segmentation(gamma_corrected, k_value)
    
    # Phase 3: Refinement
    mask = phase3_refinement(labels, original.shape)
    
    # Phase 4: Feature Extraction
    features = phase4_feature_extraction(original, mask)
    
    # Classification
    classification, classification_class = classify_durian(features)
    
    # Save processed mask image
    mask_filename = f"mask_{filename}"
    mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    
    # Create colored mask for visualization
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Apply color overlay on original
    overlay = original.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green overlay
    result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
    
    cv2.imwrite(mask_filepath, result)
    
    return jsonify({
        'success': True,
        'mask_path': f'/static/uploads/{mask_filename}',
        'features': features,
        'classification': classification,
        'classification_class': classification_class,
        'parameters': {
            'gamma': gamma,
            'k_value': k_value
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("MusangKing Classification System - GUI")
    print("=" * 50)
    print("Server starting at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
