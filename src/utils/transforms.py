import torchvision.transforms as transforms

def get_train_transforms(config):
    """
    Get the training data transforms with augmentation
    Args:
        config (dict): Configuration dictionary containing transform parameters
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomResizedCrop(
            size=config['data']['image_size'], scale=(0.8, 1.0)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['normalize']['mean'],
            std=config['augmentation']['normalize']['std']
        )
    ])

def get_val_transforms(config):
    """
    Get the validation data transforms without augmentation
    Args:
        config (dict): Configuration dictionary containing transform parameters
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['normalize']['mean'],
            std=config['augmentation']['normalize']['std']
        )
    ])