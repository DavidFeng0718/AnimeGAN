import torch

from model64 import Discriminator, Generator
from train import augment_discriminator_input, create_fixed_noise, set_requires_grad


def test_base_forward_backward_shape_and_gradients():
    generator = Generator(noise_dim=100)
    discriminator = Discriminator()
    noise = torch.randn(2, 100, 1, 1, requires_grad=True)
    generated = generator(noise)
    assert generated.shape == (2, 3, 64, 64)
    scores = discriminator(generated)
    assert scores.shape == (2,)
    scores.mean().backward()
    assert noise.grad is not None
    assert all(parameter.grad is not None for parameter in generator.parameters())
    counts = {
        "generator": sum(parameter.numel() for parameter in generator.parameters()),
        "discriminator": sum(parameter.numel() for parameter in discriminator.parameters()),
    }
    assert counts == {"generator": 1100707, "discriminator": 1437505}


def test_fixed_noise_is_independent_of_model_initialization():
    expected = create_fixed_noise(4, 100, 10042, torch.device("cpu"))
    Generator(noise_dim=100)
    actual = create_fixed_noise(4, 100, 10042, torch.device("cpu"))
    assert torch.equal(expected, actual)


def test_generator_step_does_not_accumulate_discriminator_gradients():
    generator = Generator(noise_dim=100)
    discriminator = Discriminator()
    set_requires_grad(discriminator, False)
    discriminator(generator(torch.randn(2, 100, 1, 1))).mean().backward()
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert all(parameter.grad is not None for parameter in generator.parameters())
    set_requires_grad(discriminator, True)


def test_generator_gradient_flows_through_diffaugment():
    generator = Generator(noise_dim=100)
    discriminator = Discriminator()
    set_requires_grad(discriminator, False)
    generated = generator(torch.randn(2, 100, 1, 1))
    config = {
        "diff_augment_policy": "color,translation,cutout",
        "diff_augment_translation_ratio": 0.125,
        "diff_augment_cutout_ratio": 0.5,
    }
    discriminator(augment_discriminator_input(generated, config)).mean().backward()
    assert all(parameter.grad is not None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in discriminator.parameters())
