import math

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


def get_resize_crop_steps(image_size):
    """Resize without aspect-ratio distortion, then take a square center crop."""
    return [
        transforms.Resize(
            image_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(image_size),
    ]


def get_rotation_padding(image_size, degrees):
    """Return enough context for a rotated center crop to contain no fill pixels."""
    if isinstance(degrees, (tuple, list)):
        if len(degrees) != 2:
            raise ValueError("degrees must be a number or a two-value range.")
        low_degrees, high_degrees = sorted(float(value) for value in degrees)
    else:
        max_degrees = abs(float(degrees))
        low_degrees, high_degrees = -max_degrees, max_degrees

    candidates = [low_degrees, high_degrees]
    first_peak = math.ceil((low_degrees - 45.0) / 90.0)
    last_peak = math.floor((high_degrees - 45.0) / 90.0)
    candidates.extend(45.0 + 90.0 * index for index in range(first_peak, last_peak + 1))
    max_extent = max(
        abs(math.cos(math.radians(angle))) + abs(math.sin(math.radians(angle)))
        for angle in candidates
    )
    return math.ceil(image_size * (max_extent - 1.0) / 2.0)


def get_safe_rotation_steps(image_size, degrees):
    padding = get_rotation_padding(image_size, degrees)
    steps = []
    if padding > 0:
        steps.append(transforms.Pad(padding, padding_mode="reflect"))
    steps.extend([
        transforms.RandomRotation(
            degrees,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(image_size),
    ])
    return steps


def get_transform(image_size, augmentation, options=None):
    options = options or {}
    augmentations = parse_augmentations(augmentation)

    transform_steps = get_resize_crop_steps(image_size)

    for name in augmentations:
        if name == "none":
            continue
        if name == "flip":
            transform_steps.append(transforms.RandomHorizontalFlip(
                p=options.get("horizontal_flip_probability", 0.5)
            ))
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
            rotation_probability = options.get("rotation_probability", 1.0)
            if not 0.0 <= rotation_probability <= 1.0:
                raise ValueError("rotation_probability must be in [0, 1].")
            transform_steps.append(transforms.RandomApply(
                get_safe_rotation_steps(
                    image_size,
                    options.get("rotation_degrees", 10.0),
                ),
                p=rotation_probability,
            ))
        elif name == "color_jitter":
            transform_steps.append(transforms.ColorJitter(
                brightness=options.get("color_jitter_brightness", 0.2),
                contrast=options.get("color_jitter_contrast", 0.2),
                saturation=options.get("color_jitter_saturation", 0.2),
                hue=options.get("color_jitter_hue", 0.0),
            ))
    transform_steps.append(transforms.ToTensor())
    if "gaussian_noise" in augmentations:
        transform_steps.append(transforms.Lambda(lambda tensor: add_gaussian_noise(tensor, sigma=0.02)))
    transform_steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_steps)
