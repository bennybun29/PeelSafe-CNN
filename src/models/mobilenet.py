import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BananaLeafClassifier(nn.Module):
    """EfficientNet-based classifier for banana leaf diseases"""
    def __init__(self, num_classes, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(BananaLeafClassifier, self).__init__()
        
        # Load EfficientNet with pretrained weights if specified
        if pretrained:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = efficientnet_b0(weights=None)
        
        # Replace the classifier with a new one
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

def get_model(config):
    """
    Create a model instance from config
    Args:
        config (dict): Configuration dictionary
    Returns:
        BananaLeafClassifier: Model instance
    """
    if config['model']['name'].lower() != 'efficientnet_b0':
        print(f"Warning: Config specifies {config['model']['name']} but using EfficientNet-B0")
        
    model = BananaLeafClassifier(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    return model