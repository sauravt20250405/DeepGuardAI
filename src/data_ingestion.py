import os
import numpy as np
import cv2

# Configuration
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "train")

# Ensure the target folders exist
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

def generate_mock_data():
    """Generates small, blank images so we can test the PyTorch data loaders locally."""
    print("Generating Mock Training Data (50 Real, 50 Fake)...")
    
    # We will make them 224x224 right off the bat (EfficientNet size)
    for i in range(50):
        # Create a blank black image
        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Save to the 'real' folder
        cv2.imwrite(os.path.join(REAL_DIR, f"real_test_img_{i}.jpg"), blank_img)
        # Save to the 'fake' folder
        cv2.imwrite(os.path.join(FAKE_DIR, f"fake_test_img_{i}.jpg"), blank_img)
        
    print(f"✅ Saved 50 images to: {REAL_DIR}")
    print(f"✅ Saved 50 images to: {FAKE_DIR}")
    print("\nData Pipeline is ready! You can now run train.py")

if __name__ == "__main__":
    generate_mock_data()