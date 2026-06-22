import torch
import torch.nn.functional as F


SUPPORTED_POLICIES = {"color", "translation", "cutout"}


def parse_diff_augment_policy(policy):
    if policy is None:
        return []
    if not isinstance(policy, str):
        raise TypeError("diff_augment_policy must be a comma-separated string.")

    names = [name.strip() for name in policy.split(",") if name.strip()]
    if not names or names == ["none"]:
        return []
    if "none" in names:
        raise ValueError("'none' cannot be combined with DiffAugment policies.")

    unsupported = sorted(set(names) - SUPPORTED_POLICIES)
    if unsupported:
        raise ValueError(f"Unsupported DiffAugment policies: {', '.join(unsupported)}")
    return names


def rand_brightness(images):
    offset = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device, dtype=images.dtype
    ) - 0.5
    return images + offset


def rand_saturation(images):
    image_mean = images.mean(dim=1, keepdim=True)
    scale = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device, dtype=images.dtype
    ) * 2.0
    return (images - image_mean) * scale + image_mean


def rand_contrast(images):
    image_mean = images.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device, dtype=images.dtype
    ) + 0.5
    return (images - image_mean) * scale + image_mean


def rand_color(images):
    images = rand_brightness(images)
    images = rand_saturation(images)
    return rand_contrast(images)


def rand_translation(images, ratio=0.125):
    height, width = images.shape[-2:]
    pad_y = int(round(height * ratio))
    pad_x = int(round(width * ratio))
    if pad_y == 0 and pad_x == 0:
        return images

    padded = F.pad(images, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    batch, channels = images.shape[:2]
    shift_y = torch.randint(
        -pad_y, pad_y + 1, (batch, 1, 1, 1), device=images.device
    )
    shift_x = torch.randint(
        -pad_x, pad_x + 1, (batch, 1, 1, 1), device=images.device
    )

    row_indices = (
        torch.arange(height, device=images.device).view(1, 1, height, 1)
        + pad_y
        + shift_y
    )
    row_indices = row_indices.expand(batch, channels, height, padded.shape[-1])
    translated = padded.gather(2, row_indices)

    column_indices = (
        torch.arange(width, device=images.device).view(1, 1, 1, width)
        + pad_x
        + shift_x
    )
    column_indices = column_indices.expand(batch, channels, height, width)
    return translated.gather(3, column_indices)


def rand_cutout(images, ratio=0.5):
    height, width = images.shape[-2:]
    cutout_height = int(round(height * ratio))
    cutout_width = int(round(width * ratio))
    if cutout_height == 0 or cutout_width == 0:
        return images

    cutout_height = min(cutout_height, height)
    cutout_width = min(cutout_width, width)
    batch = images.shape[0]
    top = torch.randint(
        0, height - cutout_height + 1, (batch, 1, 1, 1), device=images.device
    )
    left = torch.randint(
        0, width - cutout_width + 1, (batch, 1, 1, 1), device=images.device
    )
    rows = torch.arange(height, device=images.device).view(1, 1, height, 1)
    columns = torch.arange(width, device=images.device).view(1, 1, 1, width)
    keep_rows = (rows < top) | (rows >= top + cutout_height)
    keep_columns = (columns < left) | (columns >= left + cutout_width)
    mask = keep_rows | keep_columns
    return images * mask.to(dtype=images.dtype)


def diff_augment(images, policy="none", translation_ratio=0.125, cutout_ratio=0.5):
    policies = parse_diff_augment_policy(policy)
    if not policies:
        return images
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("DiffAugment expects a floating-point NCHW tensor.")
    if not 0.0 <= translation_ratio <= 0.5:
        raise ValueError("translation_ratio must be in [0, 0.5].")
    if not 0.0 <= cutout_ratio <= 1.0:
        raise ValueError("cutout_ratio must be in [0, 1].")

    for name in policies:
        if name == "color":
            images = rand_color(images)
        elif name == "translation":
            images = rand_translation(images, translation_ratio)
        elif name == "cutout":
            images = rand_cutout(images, cutout_ratio)
    return images.contiguous()
