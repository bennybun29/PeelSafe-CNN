import os
import argparse
import yaml
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.mobilenet import get_model
from models.verification import load_verifier, verify_image

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_transform(config):
    """Get transform pipeline for prediction"""
    return transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['normalize']['mean'],
            std=config['augmentation']['normalize']['std']
        )
    ])

def load_model(config, model_path, device):
    """Load trained model"""
    model = get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, verifier, transform, device, classes):
    """Predict disease class for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # First verify if it's a banana leaf
    is_banana_leaf, verification_confidence = verify_image(image_tensor[0], verifier)
    
    if not is_banana_leaf:
        return {
            'is_banana_leaf': False,
            'verification_confidence': verification_confidence * 100,
            'message': 'The image does not appear to contain a banana leaf.'
        }
    
    # Make disease prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()
    
    return {
        'is_banana_leaf': True,
        'verification_confidence': verification_confidence * 100,
        'class': classes[predicted_idx],
        'confidence': confidence * 100,
        'probabilities': {
            cls: prob.item() * 100 
            for cls, prob in zip(classes, probabilities)
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Predict banana leaf diseases')
    parser.add_argument('input_path', help='Path to image file or directory')
    parser.add_argument('--model_path', default='results/best_model.pth',
                      help='Path to trained model weights')
    parser.add_argument('--verifier_path', default='results/verifier_model.pth',
                      help='Path to trained verifier model weights')
    parser.add_argument('--config_path', default='configs/config.yaml',
                      help='Path to config file')
    parser.add_argument('--verification_threshold', type=float, default=0.5,
                      help='Confidence threshold for banana leaf verification')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data transform
    transform = get_transform(config)

    # Load disease classification model
    model = load_model(config, args.model_path, device)
    
    # Load verification model
    verifier = load_verifier(args.verifier_path, device)

    # Get class names from training data directory
    train_dir = config['data']['train_data_path']
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    # Process input path
    if os.path.isfile(args.input_path):
        # Single image prediction
        result = predict_image(args.input_path, model, verifier, transform, device, classes)
        print(f"\nPrediction for {os.path.basename(args.input_path)}:")
        
        if not result['is_banana_leaf']:
            print(f"Verification confidence: {result['verification_confidence']:.2f}%")
            print(result['message'])
        else:
            print(f"Verification confidence: {result['verification_confidence']:.2f}%")
            print(f"Disease Class: {result['class']}")
            print(f"Disease Confidence: {result['confidence']:.2f}%")
            print("\nClass probabilities:")
            for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"{cls}: {prob:.2f}%")
    
    else:
        # Directory prediction
        if not os.path.isdir(args.input_path):
            print(f"Error: {args.input_path} is not a valid file or directory")
            return

        print(f"\nProcessing images in {args.input_path}")
        image_files = [
            f for f in os.listdir(args.input_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        for img_file in image_files:
            img_path = os.path.join(args.input_path, img_file)
            result = predict_image(img_path, model, verifier, transform, device, classes)
            print(f"\nPrediction for {img_file}:")
            
            if not result['is_banana_leaf']:
                print(f"Verification confidence: {result['verification_confidence']:.2f}%")
                print(result['message'])
            else:
                print(f"Verification confidence: {result['verification_confidence']:.2f}%")
                print(f"Disease Class: {result['class']}")
                print(f"Disease Confidence: {result['confidence']:.2f}%")

if __name__ == '__main__':
    main()