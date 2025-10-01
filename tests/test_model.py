import unittest
import torch
import yaml
import os
from PIL import Image
import numpy as np
from torchvision import transforms

# Add the src directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mobilenet import BananaLeafClassifier, get_model
from src.data.dataset import BananaLeafDataset
from src.utils.transforms import get_train_transforms, get_val_transforms

class TestBananaLeafModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        # Load config
        with open('configs/config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Initialize model
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = get_model(cls.config)
        cls.model = cls.model.to(cls.device)
        
        # Create dummy input
        cls.batch_size = 4
        cls.channels = 3
        cls.height, cls.width = cls.config['data']['image_size']
        cls.num_classes = cls.config['model']['num_classes']

    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsInstance(self.model, BananaLeafClassifier)
        self.assertEqual(self.model.model.classifier[-1].out_features, self.num_classes)

    def test_model_forward_pass(self):
        """Test if model forward pass works with correct input shape"""
        dummy_input = torch.randn(self.batch_size, self.channels, self.height, self.width)
        dummy_input = dummy_input.to(self.device)
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check if output contains valid probabilities after softmax
        probs = torch.nn.functional.softmax(output, dim=1)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(self.batch_size).to(self.device)))

    def test_train_transforms(self):
        """Test if training transforms are working correctly"""
        transform = get_train_transforms(self.config)
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transformed_image = transform(dummy_image)
        
        self.assertEqual(transformed_image.shape, (self.channels, self.height, self.width))
        self.assertTrue(isinstance(transformed_image, torch.Tensor))

    def test_val_transforms(self):
        """Test if validation transforms are working correctly"""
        transform = get_val_transforms(self.config)
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transformed_image = transform(dummy_image)
        
        self.assertEqual(transformed_image.shape, (self.channels, self.height, self.width))
        self.assertTrue(isinstance(transformed_image, torch.Tensor))

    def test_dataset_loading(self):
        """Test if dataset class works correctly"""
        if os.path.exists(self.config['data']['train_data_path']):
            transform = get_train_transforms(self.config)
            dataset = BananaLeafDataset(self.config['data']['train_data_path'], transform=transform)
            
            # Test if dataset has correct number of classes
            self.assertEqual(len(dataset.classes), self.num_classes)
            
            # Test if we can load an image and its label
            if len(dataset) > 0:
                image, label = dataset[0]
                self.assertEqual(image.shape, (self.channels, self.height, self.width))
                self.assertTrue(isinstance(label, int))
                self.assertTrue(0 <= label < self.num_classes)

    def test_model_input_size(self):
        """Test if model accepts different batch sizes"""
        batch_sizes = [1, 2, 4, 8]
        for bs in batch_sizes:
            dummy_input = torch.randn(bs, self.channels, self.height, self.width)
            dummy_input = dummy_input.to(self.device)
            
            with torch.no_grad():
                output = self.model(dummy_input)
            
            self.assertEqual(output.shape, (bs, self.num_classes))

    def test_model_trainable_parameters(self):
        """Test if model has trainable parameters"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(trainable_params, 0)

if __name__ == '__main__':
    unittest.main()