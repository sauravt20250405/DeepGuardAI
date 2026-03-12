import os
import sys

# --- ROBUST PATH CONFIGURATION ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model_arch import DeepGuardModel

# Use CPU natively 
device = torch.device('cpu') 

# Directory Configuration
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "train")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8 # Lowered for CPU 
EPOCHS = 3
LEARNING_RATE = 0.001

def train():
    print(f"Loading data from {DATA_DIR}...")
    
    try:
        train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please run data_ingestion.py first to prepare the dataset.")
        sys.exit(1)

    print(f"Initializing EfficientNet Model on {device}...")
    model = DeepGuardModel(pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    print("Starting Training Loop...")
    print("-" * 30)

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            train_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches: 
               print(f"  [Epoch {epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{total_batches}] - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / total_batches

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        print(f"Evaluating Validation Set for Epoch {epoch+1}...")
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print("\n" + "=" * 40)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Training Loss:   {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Accuracy:        {accuracy:.2f}%")
        print("=" * 40 + "\n")

        # Save Checkpoint if validation improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODELS_DIR, "deepguard_best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to: {save_path}")

    print("Training Pipeline Complete!")

if __name__ == '__main__':
    train()