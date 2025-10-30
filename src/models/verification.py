import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F

class BananaLeafVerifier(nn.Module):
    """Binary classifier to verify if an image contains a banana leaf"""
    def __init__(self, pretrained=True):
        super(BananaLeafVerifier, self).__init__()
        
        # Use EfficientNet-B0 as the base model (smaller and faster than MobileNetV2)
        if pretrained:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = efficientnet_b0(weights=None)
        
        # Replace classifier with binary classification head
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1)  # Binary classification
        )
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))

def load_verifier(model_path=None, device='cuda'):
    """
    Load the banana leaf verification model
    Args:
        model_path (str): Path to the trained verifier model weights
        device (str): Device to load the model on
    Returns:
        BananaLeafVerifier: Loaded model
    """
    model = BananaLeafVerifier(pretrained=True)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def verify_image(image_tensor, model, threshold=0.5):
    """
    Verify if an image contains a banana leaf
    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor
        model (BananaLeafVerifier): Verification model
        threshold (float): Confidence threshold for positive classification
    Returns:
        tuple: (is_banana_leaf, confidence)
    """
    with torch.no_grad():
        confidence = model(image_tensor.unsqueeze(0)).item()
        is_banana_leaf = confidence >= threshold
    return is_banana_leaf, confidence