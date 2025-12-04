#!/usr/bin/env python3
"""
Diagnostic script to verify Face Recognition project setup
"""

import os
import sys

print("=" * 60)
print("Face Recognition Project - Diagnostics")
print("=" * 60)

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Check 1: Required directories
print("\n[1] Checking required directories...")
required_dirs = [
    "dataset",
    "face_detection_model",
    "Others",
    "output"
]

for dir_name in required_dirs:
    dir_path = os.path.join(script_dir, dir_name)
    if os.path.exists(dir_path):
        print(f"    ✓ {dir_name}/ exists")
    else:
        print(f"    ✗ {dir_name}/ NOT FOUND")

# Check 2: Required model files
print("\n[2] Checking required model files...")
model_files = {
    "face_detection_model/deploy.prototxt": "Face Detection Prototype",
    "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel": "Face Detection Model",
    "Others/openface_nn4.small2.v1.t7": "Face Embedding Model"
}

for file_path, description in model_files.items():
    full_path = os.path.join(script_dir, file_path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path) / (1024 * 1024)  # Convert to MB
        print(f"    ✓ {description} ({size:.1f} MB)")
    else:
        print(f"    ✗ {description} NOT FOUND: {file_path}")

# Check 3: Dataset images
print("\n[3] Checking dataset images...")
dataset_dir = os.path.join(script_dir, "dataset")
if os.path.exists(dataset_dir):
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        print(f"    ✓ Found {len(image_files)} training images")
        # Extract unique people names
        names = set()
        for img in image_files:
            name = img.split('-')[-1].split('.')[0]
            names.add(name)
        print(f"    ✓ Unique people in dataset: {', '.join(sorted(names))}")
    else:
        print("    ✗ No image files found in dataset/")
else:
    print("    ✗ dataset/ directory not found")

# Check 4: Output files (generated during training)
print("\n[4] Checking generated output files...")
output_files = {
    "output/embeddings.pickle": "Face Embeddings",
    "output/recognizer": "Trained SVM Model",
    "output/le.pickle": "Label Encoder"
}

all_trained = True
for file_path, description in output_files.items():
    full_path = os.path.join(script_dir, file_path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path) / (1024)  # Convert to KB
        print(f"    ✓ {description} ({size:.1f} KB)")
    else:
        print(f"    ✗ {description} NOT FOUND: {file_path}")
        all_trained = False

# Check 5: Python dependencies
print("\n[5] Checking Python dependencies...")
required_modules = {
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'imutils': 'imutils',
    'sklearn': 'scikit-learn',
    'pickle': 'pickle (built-in)'
}

missing_modules = []
for module, name in required_modules.items():
    try:
        __import__(module)
        print(f"    ✓ {name}")
    except ImportError:
        print(f"    ✗ {name} NOT INSTALLED")
        missing_modules.append(module)

# Check 6: Python version
print("\n[6] Python version check...")
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"    Python {python_version}")
if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
    print("    ✓ Version is compatible (3.7+)")
else:
    print("    ✗ Please upgrade to Python 3.7 or higher")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if all_trained:
    print("✓ All files present! Ready to run recognition.")
    print("\nRun: python recognize_video_fixed.py")
else:
    print("✗ Training files not found. Need to train the model first.")
    print("\nRun these commands in order:")
    print("  1. python extract_embeddings.py")
    print("  2. python train_model.py")
    print("  3. python recognize_video_fixed.py")

if missing_modules:
    print(f"\n✗ Missing Python modules: {', '.join(missing_modules)}")
    print(f"\nInstall with: pip install -r requirements.txt")
else:
    print("\n✓ All dependencies installed!")

print("=" * 60)
