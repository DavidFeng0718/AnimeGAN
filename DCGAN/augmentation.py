from PIL import ImageFilter
from torchvision import transforms


SUPPORTED_AUGMENTATIONS = {
    "none",
    "flip",
    "brightness",
    "contrast",
    "blur",
    "sharpen",
}


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
        transform_steps.append(transforms.RandomHorizontalFlip(p=1.0))
    elif augmentation == "brightness":
        transform_steps.append(transforms.ColorJitter(brightness=0.2))
    elif augmentation == "contrast":
        transform_steps.append(transforms.ColorJitter(contrast=0.2))
    elif augmentation == "blur":
        transform_steps.append(transforms.GaussianBlur(kernel_size=3))
    elif augmentation == "sharpen":
        transform_steps.append(transforms.Lambda(lambda image: image.filter(ImageFilter.SHARPEN)))

    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(transform_steps)
