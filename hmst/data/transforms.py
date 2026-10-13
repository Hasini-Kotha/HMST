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


def get_train_transforms(
    image_size: int = 224,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    degrees: float = 10.0,
) -> T.Compose:
    """
    Training transforms.
    Resize + horizontal flip + vertical flip + small rotation + ImageNet normalization.
    """
    transforms = [T.Resize((image_size, image_size))]
    if hflip_p > 0:
        transforms.append(T.RandomHorizontalFlip(p=hflip_p))
    if vflip_p > 0:
        transforms.append(T.RandomVerticalFlip(p=vflip_p))
    if degrees > 0:
        transforms.append(T.RandomRotation(degrees=degrees))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return T.Compose(transforms)


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
