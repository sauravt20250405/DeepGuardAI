import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
from torchvision import models

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DeepGaurdAI", "data", "processed", "train")
MODELS_DIR = os.path.join(BASE_DIR, "DeepGaurdAI", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Ensure data directories exist
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

print(f"Data Source Directory: {DATA_DIR}")

# --- 1. DATA INGESTION UTILITY ---
def ensure_mock_data():
    real_files = glob.glob(os.path.join(REAL_DIR, '*.jpg'))
    fake_files = glob.glob(os.path.join(FAKE_DIR, '*.jpg'))
    
    if len(real_files) < 10 or len(fake_files) < 10:
        print("Generating mock data for testing...")
        for i in range(50):
            blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(REAL_DIR, f"real_test_{i}.jpg"), blank_img)
            cv2.imwrite(os.path.join(FAKE_DIR, f"fake_test_{i}.jpg"), blank_img)
        print("Mock data generated.")
    else:
        print("Mock data found. Skipping generation.")

# --- 2. DATASET DEFINITION ---
class SimpleDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Simple normalization to replace Albumentations for this test
        image = image.astype(np.float32) / 255.0
        # Transpose to CHW format expected by PyTorch
        image = np.transpose(image, (2, 0, 1))
        
        # Standard normalization values
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return torch.tensor(image, dtype=torch.float32), label

def get_simple_dataloaders(batch_size=8, val_split=0.2):
    fake_paths = glob.glob(os.path.join(FAKE_DIR, '*.jpg'))
    real_paths = glob.glob(os.path.join(REAL_DIR, '*.jpg'))
    
    all_paths = fake_paths + real_paths
    all_labels = [0] * len(fake_paths) + [1] * len(real_paths)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=val_split, random_state=42, stratify=all_labels
    )

    train_dataset = SimpleDataset(train_paths, train_labels)
    val_dataset = SimpleDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

# --- 3. MODEL ARCHITECTURE ---
class SimpleDeepGuardModel(nn.Module):
    def __init__(self):
        super(SimpleDeepGuardModel, self).__init__()
        # Use a very lightweight model for local CPU testing
        self.backbone = models.mobilenet_v2(pretrained=True)
        # Replace the final layer for 2 classes (Real vs Fake)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)

    def forward(self, x):
        return self.backbone(x)

# --- 4. TRAINING LOOP ---
def train():
    ensure_mock_data()
    
    device = torch.device('cpu') 
    print(f"Device set to: {device}")
    
    train_loader, val_loader = get_simple_dataloaders()
    print("Data loaders initialized.")

    model = SimpleDeepGuardModel().to(device)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    EPOCHS = 3
    best_val_loss = float('inf')

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            train_loss += loss.item()
            print(f"Epoch {epoch+1} - Batch {batch_idx+1} Loss: {loss.item():.4f}")

        # Basic Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODELS_DIR, "deepguard_best_model.pth")
            torch.save(model.state_dict(), save_path)
            print("  [*] Model saved.")

    print("Training Complete!")

if __name__ == '__main__':
    train()