import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob

class DeepGuardDataset(Dataset):
    """
    Production-grade Dataset utilizing OpenCV for fast disk I/O
    and Albumentations for real-world data corruption during training.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Fast read using OpenCV (Loads in BGR format)
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
            
        # Convert BGR to RGB (Required for EfficientNet)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def get_dataloaders(data_dir, batch_size=32, val_split=0.2):
    """
    Builds the aggressive training pipeline and returns DataLoaders.
    """
    print("Building Production Data Pipeline with Albumentations...")
    
    # Locate all processed images
    fake_dir = os.path.join(data_dir, 'fake')
    real_dir = os.path.join(data_dir, 'real')
    
    # Support both .jpg and .png
    fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg')) + glob.glob(os.path.join(fake_dir, '*.png'))
    real_paths = glob.glob(os.path.join(real_dir, '*.jpg')) + glob.glob(os.path.join(real_dir, '*.png'))
    
    all_paths = fake_paths + real_paths
    # Labels: Fake = 0, Real = 1
    all_labels = [0] * len(fake_paths) + [1] * len(real_paths)
    
    if len(all_paths) == 0:
         raise RuntimeError(f"No images found in {data_dir}. Ensure preprocessing completed.")

    # Stratified split ensures an equal ratio of fake/real in train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=val_split, random_state=42, stratify=all_labels
    )

    # --- THE AUGMENTATION PIPELINE ---
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        
        # Simulate video compression (e.g., WhatsApp, Social Media)
        A.ImageCompression(quality_lower=50, quality_upper=85, p=0.4),
        
        # Simulate camera noise, motion, and bad focus
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Defocus(radius=(3, 5), alias_blur=(0.1, 0.5), p=1.0)
        ], p=0.3),
        
        # Simulate varying lighting conditions
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        
        # Standard Normalization for EfficientNet
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Validation must remain clean to accurately test performance
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = DeepGuardDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DeepGuardDataset(val_paths, val_labels, transform=val_transform)

    # Note: Setting num_workers=0 is often safer on Windows to avoid threading errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Total Images Found: {len(all_paths)}")
    print(f"Training Set: {len(train_dataset)} images (Augmented)")
    print(f"Validation Set: {len(val_dataset)} images (Clean)")
    print("Classes: {'fake': 0, 'real': 1}")

    return train_loader, val_loader

if __name__ == "__main__":
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "train")
    
    # Run a quick test
    try:
        train_loader, val_loader = get_dataloaders(PROCESSED_DIR)
        images, labels = next(iter(train_loader))
        print(f"Batch Image Shape: {images.shape}")
        print("Data pipeline test successful!")
    except Exception as e:
        print(f"Test failed: {e}")