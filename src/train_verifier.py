import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
from models.verification import BananaLeafVerifier

class BananaLeafVerificationDataset(Dataset):
    def __init__(self, banana_leaf_dir, negative_dir, transform=None):
        self.transform = transform
        
        # Get banana leaf images (positive samples)
        self.positive_samples = []
        for class_dir in os.listdir(banana_leaf_dir):
            class_path = os.path.join(banana_leaf_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.positive_samples.append((
                            os.path.join(class_path, img_name),
                            1  # positive label
                        ))
        
        # Get negative samples
        self.negative_samples = []
        if os.path.exists(negative_dir):
            for img_name in os.listdir(negative_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.negative_samples.append((
                        os.path.join(negative_dir, img_name),
                        0  # negative label
                    ))
        
        # Balance the dataset
        if len(self.negative_samples) > len(self.positive_samples):
            self.negative_samples = random.sample(self.negative_samples, len(self.positive_samples))
        
        self.samples = self.positive_samples + self.negative_samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_verifier(banana_leaf_dir, negative_dir, save_path, num_epochs=10, batch_size=32):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = BananaLeafVerificationDataset(banana_leaf_dir, negative_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = BananaLeafVerifier(pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print('Model saved!')
        
        print('-' * 50)

if __name__ == '__main__':
    # Paths
    banana_leaf_dir = 'data/train'  # Directory containing banana leaf images
    negative_dir = 'data/negative'   # Directory containing non-banana leaf images
    save_path = 'results/verifier_model.pth'
    
    os.makedirs('results', exist_ok=True)
    train_verifier(banana_leaf_dir, negative_dir, save_path)