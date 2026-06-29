from torch.utils.data import DataLoader, random_split
from data.dataset import AIDataset
from data.transforms import get_train_transforms, get_val_transforms


def get_dataloaders(root, batch_size=32, num_workers=0,
                    train_ratio=0.8, image_size=224):
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    full_dataset = AIDataset(root, transform=train_transform)

    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms to validation subset
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader