import pandas as pd
import joblib
import numpy as np

# Load training data
df = pd.read_csv('trained_models_v2/TRAINING_MODEL/features.csv')

print("="*60)
print("FEATURE ANALYSIS BY VARIETY")
print("="*60)

for variety in df['Variety'].unique():
    v_df = df[df['Variety'] == variety]
    print(f"\n{variety} ({len(v_df)} samples)")
    print(f"  Compactness: {v_df['Compactness'].mean():.1f} (std: {v_df['Compactness'].std():.1f})")
    print(f"  Mean_Red: {v_df['Mean_Red'].mean():.1f} (std: {v_df['Mean_Red'].std():.1f})")
    print(f"  Mean_Hue: {v_df['Mean_Hue'].mean():.1f}")

print("\n" + "="*60)
print("MODEL PREDICTION TEST")
print("="*60)

# Load model
m = joblib.load('TRAINING MODEL/variety_model.pkl')
e = joblib.load('TRAINING MODEL/variety_model_encoder.pkl')

print(f"Classes: {e.classes_}")

# Test with real image features
test_cases = [
    [139.36, 0.00516, 0.99, 0.71, 198.4],  # Real image 1
    [58.22, 0.00397, 1.2, 0.65, 150],       # Real image 2
    [130, 0.015, 1.0, 0.7, 120],            # Low Mean_Red
    [130, 0.015, 1.0, 0.7, 200],            # High Mean_Red
]

cols = ['Compactness', 'Smoothness', 'Aspect_Ratio', 'Rectangularity', 'Mean_Red']
for i, feats in enumerate(test_cases):
    X = pd.DataFrame([feats], columns=cols)
    pred = m.predict(X)[0]
    proba = m.predict_proba(X)[0]
    name = e.inverse_transform([pred])[0]
    probs = {e.classes_[j]: f"{proba[j]*100:.1f}%" for j in range(len(e.classes_))}
    print(f"\nTest {i+1}: {feats}")
    print(f"  -> {name}")
    print(f"  -> {probs}")
