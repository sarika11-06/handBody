# Don't need modification here

import os
import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths for dataset and model storage
DATASET_DIR = './dataset'
MODEL_DIR = './models'

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_model(data_file, model_name):
    print(f"🔄 Loading {model_name} dataset...")
    
    # Load data
    data_dict = pickle.load(open(os.path.join(DATASET_DIR, data_file), 'rb'))
    
    # Ensure all samples have valid features
    valid_data = []
    valid_labels = []

    for i, sample in enumerate(data_dict['data']):
        if len(sample) > 0:  # Ensure sample is not empty
            valid_data.append(sample)
            valid_labels.append(data_dict['labels'][i])

    # Convert to NumPy arrays
    data = np.asarray(valid_data)
    labels = np.asarray(valid_labels)

    print(f"✅ {len(valid_data)} valid samples found for {model_name}.")
    
    # Check label distribution
    label_counts = Counter(labels)
    print(f"📊 Class distribution: {label_counts}")

    # Find classes with fewer than 2 samples
    min_samples = min(label_counts.values())

    if min_samples < 2:
        print("⚠️ Warning: Some classes have fewer than 2 samples. Disabling stratified split.")
        stratify = None
    else:
        stratify = labels

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=stratify)

    # Train the model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Predict and check accuracy
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    print(f"🎯 {model_name} model accuracy: {score * 100:.2f}%")

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.p")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model}, f)

    print(f"💾 {model_name} model saved successfully at {model_path}.\n")

# Train models for hands and body separately
train_model('data_hands.pickle', 'model_hands')
train_model('data_body.pickle', 'model_body')

print("✅ Training process complete!")
