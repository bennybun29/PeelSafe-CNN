import torchvision.transforms as transforms

def get_train_transforms(config):
    """
    Get the training data transforms with augmentation
    Args:
        config (dict): Configuration dictionary containing transform parameters
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    transform_list = [
        transforms.Resize(config['data']['image_size']),
        transforms.RandomHorizontalFlip() if config['augmentation']['random_flip'] else None,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['normalize']['mean'],
            std=config['augmentation']['normalize']['std']
        )
    ]
    # Remove None values from the list
    transform_list = [t for t in transform_list if t is not None]
    return transforms.Compose(transform_list)

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