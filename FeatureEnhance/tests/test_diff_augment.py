import torch

from diff_augment import diff_augment, parse_diff_augment_policy


def test_none_policy_is_an_exact_noop():
    images = torch.randn(2, 3, 64, 64)
    assert diff_augment(images, "none") is images
    assert parse_diff_augment_policy("") == []


def test_full_policy_preserves_shape_dtype_device_and_gradients():
    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    augmented = diff_augment(images, "color,translation,cutout")
    assert augmented.shape == images.shape
    assert augmented.dtype == images.dtype
    assert augmented.device == images.device
    augmented.mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_translation_uses_reflection_instead_of_black_padding():
    images = torch.full((4, 3, 64, 64), 0.75, requires_grad=True)
    translated = diff_augment(images, "translation")
    assert torch.equal(translated, images)

