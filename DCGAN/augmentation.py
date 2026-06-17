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


def add_gaussian_noise(tensor, sigma=0.02):
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)


def get_transform(image_size, augmentation):
    if augmentation not in SUPPORTED_AUGMENTATIONS:
        raise ValueError(
            f"Unsupported augmentation '{augmentation}'. "
            f"Choose from: {', '.join(sorted(SUPPORTED_AUGMENTATIONS))}"
        )

    transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]

    if augmentation == "flip":
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    elif augmentation == "brightness":
        transform_steps.append(transforms.ColorJitter(brightness=0.2))
    elif augmentation == "contrast":
        transform_steps.append(transforms.ColorJitter(contrast=0.2))
    elif augmentation == "blur":
        transform_steps.append(transforms.GaussianBlur(kernel_size=3))
    elif augmentation == "sharpen":
        transform_steps.append(transforms.Lambda(lambda image: image.filter(ImageFilter.SHARPEN)))
    elif augmentation == "noise_filter":
        transform_steps.append(transforms.Lambda(lambda image: image.filter(ImageFilter.MedianFilter(size=3))))
    elif augmentation == "auto_contrast":
        transform_steps.append(transforms.Lambda(lambda image: ImageOps.autocontrast(image)))
    elif augmentation == "rotation":
        transform_steps.append(transforms.RandomRotation(10))
    elif augmentation == "color_jitter":
        transform_steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    transform_steps.append(transforms.ToTensor())
    if augmentation == "gaussian_noise":
        transform_steps.append(transforms.Lambda(lambda tensor: add_gaussian_noise(tensor, sigma=0.02)))
    transform_steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_steps)
