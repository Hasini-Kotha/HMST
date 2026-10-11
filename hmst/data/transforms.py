"""
data/transforms.py
------------------
Preprocessing transforms for Architecture 1 (HMST).

From the documents:
  - Image resize to 224x224 (standard UIQA practice, §4.2)
  - Multi-scale patch extraction happens INSIDE the model (§4.2)
  - Standard ImageNet normalization for the ResNet-50 backbone
"""

import torch
import torchvision.transforms as T


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Training transforms.
    Resize + horizontal flip + vertical flip + small rotation + ImageNet normalization.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Validation / test transforms.
    Resize + ImageNet normalization (no flip).
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
