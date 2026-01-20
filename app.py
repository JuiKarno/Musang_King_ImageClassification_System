from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import uuid
import joblib
import pandas as pd
from rembg import remove
from PIL import Image
import xgboost as xgb # Required for loading the new ensemble model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['MODEL_FOLDER'] = 'TRAINING MODEL'  # Look in TRAINING MODEL folder

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for last processing result (for processing stages page)
LAST_PROCESSING_RESULT = {}

# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================
def load_models():
    models = {}
    model_folder = app.config['MODEL_FOLDER']
    try:
        # Load models from the 'TRAINING MODEL' folder
        models['variety_model'] = joblib.load(os.path.join(model_folder, 'variety_model.pkl'))
        models['variety_encoder'] = joblib.load(os.path.join(model_folder, 'variety_model_encoder.pkl'))
        models['ripeness_model'] = joblib.load(os.path.join(model_folder, 'ripeness_model.pkl'))
        models['ripeness_encoder'] = joblib.load(os.path.join(model_folder, 'ripeness_model_encoder.pkl'))
        print("✅ Models loaded successfully!")
        return models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

MODELS = load_models()

# ============================================================
# 2. IMAGE PROCESSING FUNCTIONS (The "Syllabus" Logic)
# ============================================================

def get_mask_lab_method(image):
    """
    Standard Academic Method: Gamma -> LAB -> K-Means -> Morph
    CORRECTED to match Training Data (Gamma=0.6, K=3)
    """
    debug_images = {}
    
    # Resize to standard for consistent processing
    img_resized = cv2.resize(image, (512, 512))

    # 1. Preprocessing: Gamma Correction
    # --- FIX APPLIED: Changed from 1.2 to 0.6 to match Training Data ---
    gamma = 0.6
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_resized, table)
    debug_images['gamma'] = img_gamma

    # 2. Color Space Transformation: LAB
    img_lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2Lab)
    debug_images['lab'] = img_lab

    # 3. K-Means Clustering
    # --- FIX APPLIED: Changed K from 2 to 3 to separate Husk vs Flesh vs Background ---
    Z = img_lab.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3 
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img_lab.shape))
    debug_images['kmeans'] = cv2.cvtColor(result_image, cv2.COLOR_Lab2BGR) # Convert back for display

    # 4. Extract Mask (Assume center pixel is fruit)
    h, w = result_image.shape[:2]
    center_color = result_image[h//2, w//2]
    lower = np.array(center_color, dtype="uint8")
    upper = np.array(center_color, dtype="uint8")
    mask = cv2.inRange(result_image, lower, upper)

    # 5. Morphological Operations (Closing)
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug_images['morphology'] = mask_closed # Save intermediate

    # 6. Contour Detection & Selection
    cnts = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    
    mask_final = np.zeros_like(mask)
    contour_img = img_resized.copy() # For visualization
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        # Draw all contours in red, largest in green
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1) 
        cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)
        
        # Create final mask from largest contour
        cv2.drawContours(mask_final, [c], -1, 255, -1)
        
    debug_images['contour'] = contour_img
    debug_images['binary_mask'] = mask_final
    
    return mask_final, "Hybrid (K-Means)", debug_images
    


def get_mask_ai_method(image_path):
    # Uses 'rembg' library (U-2 Net model)
    try:
        with open(image_path, 'rb') as i:
            input_data = i.read()
            output_data = remove(input_data)
            
        # Convert back to cv2 image
        nparr = np.frombuffer(output_data, np.uint8)
        img_bg_removed = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # Extract Alpha channel as mask
        if img_bg_removed.shape[2] == 4:
            mask = img_bg_removed[:, :, 3] 
        else:
            mask = np.zeros(img_bg_removed.shape[:2], dtype=np.uint8)
            
        # Resize AI mask to 512x512 for consistency with feature extraction
        return cv2.resize(mask, (512, 512))
    except Exception as e:
        print(f"AI Mask Failed: {e}")
        return np.zeros((512, 512), dtype=np.uint8)

def smart_segmentation(image_path):
    # STRATEGY: Academic Approach (K-Means First)
    
    original = cv2.imread(image_path)
    
    # 1. Run Standard K-Means (Primary)
    print("🔄 Smart Segment: Running Standard K-Means...")
    mask_lab, _, debug_imgs = get_mask_lab_method(original)
    
    # Verify if K-Means produced a valid mask (at least 1% fruit)
    if cv2.countNonZero(mask_lab) > (original.shape[0] * original.shape[1] * 0.01):
        return mask_lab, "Standard (K-Means)", debug_imgs
        
    # 2. Fallback: AI (Only if K-Means failed)
    print("⚠️ K-Means Result Poor/Empty. Switching to AI Fallback...")
    try:
        mask_ai = get_mask_ai_method(image_path)
        if mask_ai is not None and cv2.countNonZero(mask_ai) > 0:
            return mask_ai, "Fallback (AI-Powered)", debug_imgs
    except Exception as e:
        print(f"⚠️ AI Failed: {e}")

    # If both fail, return the K-Means result (even if bad)
    return mask_lab, "Standard (K-Means)", debug_imgs


# ============================================================
# 3. FEATURE EXTRACTION (Phase 4 Logic)
# ============================================================

def phase4_feature_extraction(original_image, mask):
    img = cv2.resize(original_image, (512, 512))
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    
    if not contours: return None
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area == 0: return None

    # Geometric Features
    compactness = (perimeter ** 2) / area
    
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    smoothness = len(approx) / perimeter
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    rect_area = w * h
    rectangularity = area / rect_area if rect_area > 0 else 0
    
    # Color Features
    mean_val = cv2.mean(img, mask=mask)
    mean_red = mean_val[2] # Red channel
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    mean_hue = mean_hsv[0] # Hue channel

    # Prepare Inputs for Models (Must match training columns exactly)
    
    # Variety: [Compactness, Smoothness, Aspect_Ratio, Rectangularity, Mean_Red]
    variety_cols = ['Compactness', 'Smoothness', 'Aspect_Ratio', 'Rectangularity', 'Mean_Red']
    variety_feats = pd.DataFrame([[compactness, smoothness, aspect_ratio, rectangularity, mean_red]], columns=variety_cols)
    
    # Ripeness: [Mean_Hue, Compactness, Smoothness]
    ripeness_cols = ['Mean_Hue', 'Compactness', 'Smoothness']
    ripeness_feats = pd.DataFrame([[mean_hue, compactness, smoothness]], columns=ripeness_cols)
    
    # -----------------------------------------------------
    # VISUALIZATION 1: GEOMETRIC FEATURES (Step 6)
    # -----------------------------------------------------
    vis_geom = img.copy()
    # Draw convex hull / polygon (Green)
    cv2.drawContours(vis_geom, [approx], -1, (0, 255, 0), 2)
    # Draw bounding box (Blue)
    cv2.rectangle(vis_geom, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Add Text (Compactness & Aspect Ratio)
    cv2.putText(vis_geom, f"Compactness: {compactness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_geom, f"Aspect Ratio: {aspect_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -----------------------------------------------------
    # VISUALIZATION 2: COLOR ANALYSIS (Step 7 - Histogram)
    # -----------------------------------------------------
    # Create dark background
    vis_hist = np.zeros((300, 512, 3), dtype=np.uint8) + 30 
    
    # Draw Histograms for B, G, R
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        # Scale width by 2 to fill 512px width
        pts = np.int32(np.column_stack((np.arange(256)*2, 300 - np.int32(hist)))) 
        cv2.polylines(vis_hist, [pts], False, (255 if i==0 else 0, 255 if i==1 else 0, 255 if i==2 else 0), 2)
    
    cv2.putText(vis_hist, f"Mean Red: {mean_red:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis_hist, f"Mean Hue: {mean_hue:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return variety_feats, ripeness_feats, {
        "compactness": round(compactness, 2), 
        "smoothness": round(smoothness, 4), 
        "mean_hue": round(mean_hue, 1),
        "mean_red": round(mean_red, 1),
        "aspect_ratio": round(aspect_ratio, 2)
    }, {
        "features": vis_geom,
        "color_analysis": vis_hist
    }

def classify_durian(v_input, r_input):
    if MODELS is None: return "Model Error", "unknown", 0.0, None
    
    try:
        # DEBUG: Print feature values
        print("\n" + "="*50, flush=True)
        print("🔍 DEBUG: Feature Values Received", flush=True)
        print("="*50, flush=True)
        print(f"Variety Features:\n{v_input}", flush=True)
        print(f"Ripeness Features:\n{r_input}", flush=True)
        
        # --- VARIETY PREDICTION ---
        v_pred = MODELS['variety_model'].predict(v_input)[0]
        v_name = MODELS['variety_encoder'].inverse_transform([v_pred])[0]
        
        # Get all class probabilities
        reasoning = {}
        try:
            v_proba = MODELS['variety_model'].predict_proba(v_input)[0]
            v_classes = MODELS['variety_encoder'].classes_
            v_conf = np.max(v_proba) * 100
            
            print(f"\n📊 Variety Prediction Probabilities:", flush=True)
            proba_breakdown = {}
            for i, cls in enumerate(v_classes):
                print(f"   {cls}: {v_proba[i]*100:.1f}%", flush=True)
                # Map class names for display
                display_cls = {
                    'D175_UdangMerah': 'Udang Merah',
                    'D197_MusangKing': 'Musang King',
                    'D200_BlackThorn': 'Black Thorn'
                }.get(cls, cls)
                proba_breakdown[display_cls] = round(v_proba[i]*100, 1)
            
            reasoning['probabilities'] = proba_breakdown
            reasoning['confidence'] = round(v_conf, 1)
            
        except Exception as e:
            print(f"   Could not get probabilities: {e}", flush=True)
            v_conf = 0.0
            
        # SECURITY CHECK: Is it actually a durian?
        CONFIDENCE_THRESHOLD = 40.0
        
        if v_conf < CONFIDENCE_THRESHOLD:
            print(f"⚠️ Low Confidence ({v_conf:.1f}%). Rejecting as non-durian.")
            
            # Build detailed reasoning
            reasoning['rejected'] = True
            reasoning['threshold'] = CONFIDENCE_THRESHOLD
            
            # Analyze why it's not a durian
            reasons = []
            
            # Check if probabilities are too spread out (uncertain)
            if 'probabilities' in reasoning:
                probs = list(reasoning['probabilities'].values())
                max_prob = max(probs)
                if max_prob < 35:
                    reasons.append(f"No variety matched confidently (highest: {max_prob:.1f}%)")
                if max(probs) - min(probs) < 20:
                    reasons.append("Probabilities too similar across all classes (model is unsure)")
            
            # Feature-based reasoning
            compactness = v_input['Compactness'].values[0]
            mean_red = v_input['Mean_Red'].values[0]
            
            if compactness > 100:
                reasons.append(f"Shape too irregular (compactness: {compactness:.1f}, expected: 20-80)")
            if mean_red < 50 or mean_red > 200:
                reasons.append(f"Color outside durian range (red channel: {mean_red:.1f})")
            
            if not reasons:
                reasons.append("Object features don't match any trained durian variety")
            
            reasoning['reasons'] = reasons
            reasoning['summary'] = "This object does not appear to be a durian from our trained varieties."
            
            return "Unknown Object / Not Durian", "unknown", v_conf, reasoning

        print(f"\n🏆 Predicted Variety: {v_name} ({v_conf:.1f}% confidence)", flush=True)
        
        # --- RIPENESS PREDICTION ---
        r_pred = MODELS['ripeness_model'].predict(r_input)[0]
        r_name = MODELS['ripeness_encoder'].inverse_transform([r_pred])[0]
        
        # Get ripeness probabilities
        try:
            r_proba = MODELS['ripeness_model'].predict_proba(r_input)[0]
            r_classes = MODELS['ripeness_encoder'].classes_
            print(f"\n📊 Ripeness Prediction Probabilities:", flush=True)
            ripe_probs = {}
            for i, cls in enumerate(r_classes):
                print(f"   {cls}: {r_proba[i]*100:.1f}%", flush=True)
                ripe_probs[cls.capitalize()] = round(r_proba[i]*100, 1)
            reasoning['ripeness_probabilities'] = ripe_probs
        except:
            pass
        
        print(f"🍈 Predicted Ripeness: {r_name}", flush=True)
        print("="*50 + "\n", flush=True)
        
        # --- MAPPING ---
        css_map = {
            'D175_UdangMerah': 'udang-merah',
            'D197_MusangKing': 'musang-king',
            'D200_BlackThorn': 'black-thorn'
        }
        
        name_map = {
            'D175_UdangMerah': 'Udang Merah',
            'D197_MusangKing': 'Musang King',
            'D200_BlackThorn': 'Black Thorn'
        }
        
        display_name = name_map.get(v_name, v_name)
        css_class = css_map.get(v_name, 'unknown')
        ripeness_display = r_name.capitalize()
        
        reasoning['rejected'] = False
        reasoning['variety'] = display_name
        reasoning['ripeness'] = ripeness_display
        
        text = f"{display_name} ({ripeness_display})"
        return text, css_class, v_conf, reasoning
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", "unknown", 0.0, None


# ============================================================
# 4. FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/datasets')
def datasets():
    return render_template('datasets.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/processing_stages')
def processing_stages():
    """Display detailed processing stages for the last processed image"""
    global LAST_PROCESSING_RESULT
    return render_template('processing_stages.html', data=LAST_PROCESSING_RESULT if LAST_PROCESSING_RESULT else None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return jsonify({'filename': filename})

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        filename = data.get('filename')
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 404
        
        # 1. HYBRID SEGMENTATION (The Magic Logic)
        mask, method_name, debug_imgs = smart_segmentation(path)
        
        # 2. FEATURE EXTRACTION
        original = cv2.imread(path)
        feats = phase4_feature_extraction(original, mask)
        
        if feats is None:
            return jsonify({'error': 'No fruit detected (Mask is empty)'}), 400
            
        v_input, r_input, stats, viz_imgs = feats
        
        # Merge visualization images into debug_imgs for auto-saving
        if viz_imgs:
            debug_imgs.update(viz_imgs)
        
        # 3. CLASSIFY
        text, css, conf, reasoning = classify_durian(v_input, r_input)
        
        # 4. SAVE PIPELINE IMAGES
        pipeline_urls = {}
        timestamp = int(os.path.getmtime(path)) 
        
        # Save the main mask
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
        cv2.imwrite(mask_path, mask)
        
        # Save intermediate steps
        if debug_imgs:
            for name, img in debug_imgs.items():
                step_filename = f"{name}_{filename}"
                step_path = os.path.join(app.config['UPLOAD_FOLDER'], step_filename)
                cv2.imwrite(step_path, img)
                pipeline_urls[name] = f"/static/uploads/{step_filename}?t={timestamp}"

        # Generate final green overlay
        overlay_filename = f"overlay_{filename}"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        
        img_resized = cv2.resize(original, (512, 512))
        overlay = img_resized.copy()
        overlay[mask > 0] = [0, 255, 0] # Green
        result = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)
        cv2.imwrite(overlay_path, result)
        
        # Build response data
        response_data = {
            'success': True,
            'classification': text,
            'classification_class': css,
            'confidence': round(conf, 1),
            'features': stats,
            'mask_path': f'/static/uploads/{mask_filename}?t={timestamp}',
            'overlay_path': f'/static/uploads/{overlay_filename}?t={timestamp}',
            'original_path': f'/static/uploads/{filename}?t={timestamp}',
            'method_used': method_name,
            'pipeline': pipeline_urls,
            'reasoning': reasoning
        }
        
        # Store for processing stages page
        global LAST_PROCESSING_RESULT
        LAST_PROCESSING_RESULT = response_data
        
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("🚀 MusangKing Hybrid System Started - VERSION 2.1 (FIXED)", flush=True)
    print("   - Primary: K-Means (Lab Materials)")
    print("   - Fallback: AI (rembg)")
    print("   - Scope: Musang King, Black Thorn, Udang Merah")
    app.run(debug=True, port=5000)