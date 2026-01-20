# ============================================================
# MUSANGKING MODEL TRAINING SCRIPT
# Aligned with app.py K-Means Segmentation Pipeline
# ============================================================
# 
# This script trains a model using the EXACT SAME segmentation
# and feature extraction as your Flask app. This ensures that
# what the model learns matches what it sees at inference time.
#
# Dataset Sources:
# - https://universe.roboflow.com/durian-cf87w/durian_own/
# - https://universe.roboflow.com/carl-bwzge/durian-thesis/
# ============================================================

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

# IMPORTANT: Set your dataset path here!
# Expected structure:
#   DATASET/
#     D197_MusangKing/
#       mature/
#       immature/
#       defective/
#     D200_BlackThorn/
#       mature/
#       ...
#     D175_UdangMerah/
#       ...

DATASET_PATH = "DATASET"  # <-- CHANGE THIS TO YOUR DATASET FOLDER

# Output folder for trained models
OUTPUT_FOLDER = "TRAINING MODEL"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# STEP 2: K-MEANS SEGMENTATION (IDENTICAL TO APP.PY)
# ============================================================

def get_mask_lab_method(image):
    """
    Standard Academic Method: Gamma -> LAB -> K-Means -> Morph
    This is EXACTLY the same as app.py to ensure training/inference consistency.
    """
    # Resize to standard for consistent processing
    img_resized = cv2.resize(image, (512, 512))

    # 1. Preprocessing: Gamma Correction (SAME AS APP.PY: gamma=0.6)
    gamma = 0.6
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_resized, table)

    # 2. Color Space Transformation: LAB
    img_lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2Lab)

    # 3. K-Means Clustering (SAME AS APP.PY: K=3)
    Z = img_lab.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img_lab.shape))

    # 4. Extract Mask (Assume center pixel is fruit)
    h, w = result_image.shape[:2]
    center_color = result_image[h//2, w//2]
    lower = np.array(center_color, dtype="uint8")
    upper = np.array(center_color, dtype="uint8")
    mask = cv2.inRange(result_image, lower, upper)

    # 5. Morphological Operations (Closing)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise - keep only largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask)
        cv2.drawContours(mask_clean, [c], -1, 255, -1)
        mask = mask_clean
    
    return mask

# ============================================================
# STEP 3: FEATURE EXTRACTION (IDENTICAL TO APP.PY)
# ============================================================

def extract_features(image, mask):
    """
    Extract features EXACTLY as app.py does.
    Returns: (variety_features, ripeness_features) or None if failed
    """
    img = cv2.resize(image, (512, 512))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area == 0 or perimeter == 0:
        return None

    # Geometric Features (SAME AS APP.PY)
    compactness = (perimeter ** 2) / area
    
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    smoothness = len(approx) / perimeter
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    rect_area = w * h
    rectangularity = area / rect_area if rect_area > 0 else 0
    
    # Color Features (SAME AS APP.PY)
    mean_val = cv2.mean(img, mask=mask)
    mean_red = mean_val[2]  # Red channel
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    mean_hue = mean_hsv[0]  # Hue channel

    return {
        'Compactness': compactness,
        'Smoothness': smoothness,
        'Aspect_Ratio': aspect_ratio,
        'Rectangularity': rectangularity,
        'Mean_Red': mean_red,
        'Mean_Hue': mean_hue
    }

# ============================================================
# STEP 4: DATASET LOADING
# ============================================================

def load_dataset(dataset_path):
    """
    Load dataset and extract features using K-Means segmentation.
    
    Expected folder structure:
        DATASET/
            D197_MusangKing/
                mature/
                immature/
                defective/
            D200_BlackThorn/
                mature/
                immature/
                defective/
            D175_UdangMerah/
                mature/
                immature/
                defective/
    """
    data = []
    
    # Variety folders
    for variety_folder in os.listdir(dataset_path):
        variety_path = os.path.join(dataset_path, variety_folder)
        if not os.path.isdir(variety_path):
            continue
            
        variety_name = variety_folder  # e.g., "D197_MusangKing"
        
        # Ripeness folders
        for ripeness_folder in os.listdir(variety_path):
            ripeness_path = os.path.join(variety_path, ripeness_folder)
            if not os.path.isdir(ripeness_path):
                continue
                
            ripeness_name = ripeness_folder  # e.g., "mature"
            
            # Images
            image_files = [f for f in os.listdir(ripeness_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Processing {variety_name}/{ripeness_name}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"{variety_name}/{ripeness_name}"):
                img_path = os.path.join(ripeness_path, img_file)
                
                try:
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Get K-Means mask (SAME AS APP.PY)
                    mask = get_mask_lab_method(image)
                    
                    # Check if mask is valid (at least 1% of image)
                    if cv2.countNonZero(mask) < (512 * 512 * 0.01):
                        print(f"  Skipped (empty mask): {img_file}")
                        continue
                    
                    # Extract features (SAME AS APP.PY)
                    features = extract_features(image, mask)
                    if features is None:
                        continue
                    
                    # Add labels
                    features['Variety'] = variety_name
                    features['Ripeness'] = ripeness_name
                    features['File'] = img_file
                    
                    data.append(features)
                    
                except Exception as e:
                    print(f"  Error processing {img_file}: {e}")
                    continue
    
    return pd.DataFrame(data)

# ============================================================
# STEP 5: MODEL TRAINING
# ============================================================

def train_variety_model(df):
    """Train variety classification model (SMOTE + Ensemble)"""
    print("\n" + "="*50)
    print("TRAINING VARIETY MODEL")
    print("="*50)
    
    # Features for variety classification
    feature_cols = ['Compactness', 'Smoothness', 'Aspect_Ratio', 'Rectangularity', 'Mean_Red']
    X = df[feature_cols]
    y = df['Variety']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Classes: {encoder.classes_}")
    print(f"Class distribution: {pd.Series(y_encoded).value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # SMOTE for class balancing
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
    
    # Create ensemble model
    print("\nTraining Ensemble (SVM + RF + XGBoost)...")
    
    svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    ensemble = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('xgb', xgb_clf)],
        voting='soft'
    )
    
    ensemble.fit(X_train_smote, y_train_smote)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nVariety Model Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    return ensemble, encoder

def train_ripeness_model(df):
    """Train ripeness classification model (SMOTE + Ensemble)"""
    print("\n" + "="*50)
    print("TRAINING RIPENESS MODEL")
    print("="*50)
    
    # Features for ripeness classification
    feature_cols = ['Mean_Hue', 'Compactness', 'Smoothness']
    X = df[feature_cols]
    y = df['Ripeness']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Classes: {encoder.classes_}")
    print(f"Class distribution: {pd.Series(y_encoded).value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # SMOTE for class balancing
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
    
    # Create ensemble model
    print("\nTraining Ensemble (SVM + RF + XGBoost)...")
    
    svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    ensemble = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('xgb', xgb_clf)],
        voting='soft'
    )
    
    ensemble.fit(X_train_smote, y_train_smote)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRipeness Model Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    return ensemble, encoder

# ============================================================
# STEP 6: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("MUSANGKING ALIGNED TRAINING PIPELINE")
    print("Using K-Means Segmentation (Same as app.py)")
    print("="*60)
    
    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ ERROR: Dataset folder not found: {DATASET_PATH}")
        print("\nPlease:")
        print("1. Download images from Roboflow")
        print("2. Organize them in this structure:")
        print("   DATASET/")
        print("     D197_MusangKing/")
        print("       mature/")
        print("       immature/")
        print("       defective/")
        print("     D200_BlackThorn/")
        print("       ...")
        print("     D175_UdangMerah/")
        print("       ...")
        print(f"\n3. Update DATASET_PATH in this script to point to your folder")
        exit(1)
    
    # Step 1: Load and process dataset
    print("\n📂 Loading dataset and extracting features...")
    df = load_dataset(DATASET_PATH)
    
    if len(df) == 0:
        print("❌ No valid images found!")
        exit(1)
    
    print(f"\n✅ Processed {len(df)} images successfully!")
    print(f"\nDataset Summary:")
    print(df[['Variety', 'Ripeness']].value_counts())
    
    # Save feature CSV for reference
    csv_path = os.path.join(OUTPUT_FOLDER, "extracted_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Features saved to: {csv_path}")
    
    # Step 2: Train Variety Model
    variety_model, variety_encoder = train_variety_model(df)
    
    # Step 3: Train Ripeness Model
    ripeness_model, ripeness_encoder = train_ripeness_model(df)
    
    # Step 4: Save Models
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    
    joblib.dump(variety_model, os.path.join(OUTPUT_FOLDER, "variety_model.pkl"))
    joblib.dump(variety_encoder, os.path.join(OUTPUT_FOLDER, "variety_model_encoder.pkl"))
    joblib.dump(ripeness_model, os.path.join(OUTPUT_FOLDER, "ripeness_model.pkl"))
    joblib.dump(ripeness_encoder, os.path.join(OUTPUT_FOLDER, "ripeness_model_encoder.pkl"))
    
    print(f"✅ Models saved to: {OUTPUT_FOLDER}/")
    print("   - variety_model.pkl")
    print("   - variety_model_encoder.pkl")
    print("   - ripeness_model.pkl")
    print("   - ripeness_model_encoder.pkl")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Restart your Flask app to load the new models")
    print("2. Test the classification - it should now be accurate!")
    print("   (Because training and inference use the SAME K-Means method)")
