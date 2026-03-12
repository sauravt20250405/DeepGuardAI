import torch
import torch.nn as nn
from torchvision import models

class DeepGuardModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepGuardModel, self).__init__()
        
        # Load the EfficientNet-B0 backbone
        # We use pretrained ImageNet weights to give the model a massive head start
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # EfficientNet-B0 was originally built to classify 1,000 different objects.
        # We need to swap out its final "brain" layer to only classify 2 things: Fake (0) or Real (1).
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the final linear layer
        self.backbone.classifier[1] = nn.Linear(in_features, 2)
        
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Quick test to ensure the model compiles
    print("Initializing DeepGuard AI Model...")
    model = DeepGuardModel(pretrained=True)
    
    # Simulate the exact batch shape your dataset.py just outputted
    dummy_input = torch.randn(32, 3, 224, 224) 
    
    # Pass it through the model
    output = model(dummy_input)
    
    print("\nModel Compiled Successfully!")
    print(f"Input Shape:  {dummy_input.shape}  -> (Batch Size, Channels, Height, Width)")
    print(f"Output Shape: {output.shape}       -> (Batch Size, 2 Classes [Fake, Real])")