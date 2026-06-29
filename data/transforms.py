import torchvision.transforms as T


def get_train_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
