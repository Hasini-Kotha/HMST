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
    Resize + random horizontal flip + ImageNet normalization.
    No complex augmentation is specified in the documents,
    so only standard augmentation is applied.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
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
