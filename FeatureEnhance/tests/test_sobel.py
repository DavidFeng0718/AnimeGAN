import torch

from feature_modules import SobelFeatureExtractor, SobelResidualFusion
from model64 import Generator as BaseGenerator
from model64_sobel import Generator
from train import init_dcgan_weights, restore_training_checkpoint, save_checkpoint
from model64 import Discriminator


def test_sobel_is_fixed_differentiable_and_has_reflected_boundaries():
    sobel = SobelFeatureExtractor(3)
    assert "kernels" in dict(sobel.named_buffers())
    assert not list(sobel.parameters())

    constant = torch.full((2, 3, 16, 16), 0.75, requires_grad=True)
    edges = sobel(constant)
    assert edges.shape == (2, 6, 16, 16)
    assert edges.abs().max().item() == 0.0
    edges.sum().backward()
    assert constant.grad is not None


def test_sobel_fusion_shape_device_dtype_and_gate_identity():
    fusion = SobelResidualFusion(32, 16)
    features = torch.randn(2, 32, 32, 32, requires_grad=True)
    output = fusion(features)
    assert output.shape == features.shape
    assert output.dtype == features.dtype
    assert output.device == features.device
    assert torch.equal(output, features)
    output.mean().backward()
    assert features.grad is not None
    assert fusion.gate.grad is not None


def test_e1_preserves_shared_base_initialization_and_parameter_budget():
    torch.manual_seed(42)
    base = BaseGenerator(100)
    base.apply(init_dcgan_weights)

    torch.manual_seed(42)
    e1 = Generator(100)
    e1.apply(init_dcgan_weights)
    shared_parameters = list(e1.stem.parameters()) + list(e1.to_rgb.parameters())
    for base_parameter, e1_parameter in zip(base.parameters(), shared_parameters):
        assert torch.equal(base_parameter, e1_parameter)

    assert sum(parameter.numel() for parameter in e1.parameters()) == 1103316
    generated = e1(torch.randn(2, 100, 1, 1))
    assert generated.shape == (2, 3, 64, 64)
    generated.mean().backward()
    assert e1.edge_fusion.gate.grad is not None


def test_e1_checkpoint_round_trip(tmp_path):
    generator = Generator(100)
    discriminator = Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
    save_checkpoint(
        tmp_path, 0, generator, discriminator, g_optimizer, d_optimizer,
        {"g_loss": 1.0}, {"model_variant": "e1_sobel"},
    )
    restored_generator = Generator(100)
    restored_discriminator = Discriminator()
    restored_g_optimizer = torch.optim.Adam(restored_generator.parameters(), lr=0.0002)
    restored_d_optimizer = torch.optim.Adam(restored_discriminator.parameters(), lr=0.00005)
    start_epoch = restore_training_checkpoint(
        tmp_path / "checkpoint_epoch_1.pt",
        restored_generator,
        restored_discriminator,
        restored_g_optimizer,
        restored_d_optimizer,
        torch.device("cpu"),
    )
    assert start_epoch == 1
    for expected, actual in zip(generator.parameters(), restored_generator.parameters()):
        assert torch.equal(expected, actual)
