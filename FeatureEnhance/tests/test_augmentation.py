import numpy as np
import pytest
import torch
from PIL import Image

from augmentation import get_rotation_padding, get_transform
from create_dataset import save_img


def denormalize(tensor):
    return tensor.mul(0.5).add(0.5)


def test_rotation_uses_reflected_context_instead_of_black_fill():
    image = Image.fromarray(np.full((96, 96, 3), 180, dtype=np.uint8))
    torch.manual_seed(7)
    transformed = denormalize(get_transform(
        64,
        "rotation",
        {"rotation_degrees": 6.0, "rotation_probability": 1.0},
    )(image))

    assert transformed.shape == (3, 64, 64)
    assert transformed.min().item() > 0.69
    assert get_rotation_padding(64, 6.0) == 4
    assert get_rotation_padding(64, 90.0) == 14


def test_resize_and_center_crop_preserve_output_shape():
    image = Image.fromarray(np.full((80, 120, 3), 128, dtype=np.uint8))
    transformed = get_transform(64, "none")(image)
    assert transformed.shape == (3, 64, 64)


def test_saved_preview_uses_white_grid_separators(tmp_path):
    output = tmp_path / "preview.png"
    save_img(torch.zeros(2, 3, 64, 64), output)
    pixels = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(pixels[0, 0] == 255)


def test_rotation_probability_can_disable_rotation_and_is_validated():
    pixels = np.zeros((96, 96, 3), dtype=np.uint8)
    pixels[:, :48, 0] = 255
    image = Image.fromarray(pixels)
    expected = get_transform(64, "none")(image)
    actual = get_transform(
        64,
        "rotation",
        {"rotation_degrees": 6.0, "rotation_probability": 0.0},
    )(image)
    assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match="rotation_probability"):
        get_transform(
            64,
            "rotation",
            {"rotation_degrees": 6.0, "rotation_probability": 1.1},
        )
