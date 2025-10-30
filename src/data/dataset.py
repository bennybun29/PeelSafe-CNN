import os
from PIL import Image
from torch.utils.data import Dataset

class BananaLeafDataset(Dataset):
    """Custom Dataset for loading banana leaf disease images"""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in disease-specific folders
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def get_datasets(config, train_transform, val_transform):
    """
    Create training and validation datasets
    Args:
        config (dict): Configuration dictionary
        train_transform (callable): Transforms for training data
        val_transform (callable): Transforms for validation data
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_dataset = BananaLeafDataset(
        root_dir=config['data']['train_data_path'],
        transform=train_transform
    )
    
    val_dataset = BananaLeafDataset(
        root_dir=config['data']['val_data_path'],
        transform=val_transform
    )
    
    return train_dataset, val_dataset