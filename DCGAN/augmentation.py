import torch
from PIL import ImageFilter, ImageOps
from torchvision import transforms


SUPPORTED_AUGMENTATIONS = {
    "none",
    "flip",
    "brightness",
    "contrast",
    "blur",
    "sharpen",
    "noise_filter",
    "auto_contrast",
    "rotation",
    "color_jitter",
    "gaussian_noise",
}


def parse_augmentations(augmentation):
    if isinstance(augmentation, str):
        names = [name.strip() for name in augmentation.split("+") if name.strip()]
    elif isinstance(augmentation, (list, tuple)):
        names = list(augmentation)
    else:
        raise TypeError("augmentation must be a string or a list/tuple of strings.")

    if not names:
        names = ["none"]
    if "none" in names and len(names) > 1:
        raise ValueError("'none' cannot be combined with other augmentations.")

    unsupported = [name for name in names if name not in SUPPORTED_AUGMENTATIONS]
    if unsupported:
        raise ValueError(
            f"Unsupported augmentation(s): {', '.join(unsupported)}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_AUGMENTATIONS))}"
        )
    return names


def add_gaussian_noise(tensor, sigma=0.02):
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)


def get_transform(image_size, augmentation):
    augmentations = parse_augmentations(augmentation)

    transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]

    for name in augmentations:
        if name == "none":
            continue
        if name == "flip":
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        elif name == "brightness":
            transform_steps.append(transforms.ColorJitter(brightness=0.2))
        elif name == "contrast":
            transform_steps.append(transforms.ColorJitter(contrast=0.2))
        elif name == "blur":
            transform_steps.append(transforms.GaussianBlur(kernel_size=3))
        elif name == "sharpen":
            transform_steps.append(transforms.Lambda(lambda image: image.filter(ImageFilter.SHARPEN)))
        elif name == "noise_filter":
            transform_steps.append(transforms.Lambda(lambda image: image.filter(ImageFilter.MedianFilter(size=3))))
        elif name == "auto_contrast":
            transform_steps.append(transforms.Lambda(lambda image: ImageOps.autocontrast(image)))
        elif name == "rotation":
            transform_steps.append(transforms.RandomRotation(10))
        elif name == "color_jitter":
            transform_steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    transform_steps.append(transforms.ToTensor())
    if "gaussian_noise" in augmentations:
        transform_steps.append(transforms.Lambda(lambda tensor: add_gaussian_noise(tensor, sigma=0.02)))
    transform_steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_steps)
