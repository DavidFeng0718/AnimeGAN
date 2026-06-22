import pytest
import torch

from feature_modules import HaarDWT, HaarResidualFusion
from model64 import Discriminator, Generator as BaseGenerator
from model64_dwt import Generator
from train import init_dcgan_weights, restore_training_checkpoint, save_checkpoint


def test_haar_dwt_subbands_shape_values_and_gradients():
    dwt = HaarDWT(1)
    assert "kernels" in dict(dwt.named_buffers())
    assert not list(dwt.parameters())
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    actual = dwt(inputs)
    expected = torch.tensor([[[[5.0]], [[2.0]], [[1.0]], [[0.0]]]])
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert inputs.grad is not None


def test_haar_dwt_rejects_odd_spatial_dimensions():
    with pytest.raises(ValueError, match="even"):
        HaarDWT(3)(torch.randn(1, 3, 15, 16))


def test_haar_fusion_shape_device_dtype_and_gate_identity():
    fusion = HaarResidualFusion(32, 32)
    features = torch.randn(2, 32, 32, 32, requires_grad=True)
    output = fusion(features)
    assert output.shape == features.shape
    assert output.dtype == features.dtype
    assert output.device == features.device
    assert torch.equal(output, features)
    output.mean().backward()
    assert features.grad is not None
    assert fusion.gate.grad is not None


def test_f1_preserves_shared_base_initialization_and_parameter_budget():
    torch.manual_seed(42)
    base = BaseGenerator(100)
    base.apply(init_dcgan_weights)

    torch.manual_seed(42)
    f1 = Generator(100)
    f1.apply(init_dcgan_weights)
    shared_parameters = list(f1.stem.parameters()) + list(f1.to_rgb.parameters())
    for base_parameter, f1_parameter in zip(base.parameters(), shared_parameters):
        assert torch.equal(base_parameter, f1_parameter)

    assert sum(parameter.numel() for parameter in f1.parameters()) == 1106916
    generated = f1(torch.randn(2, 100, 1, 1))
    assert generated.shape == (2, 3, 64, 64)
    generated.mean().backward()
    assert f1.frequency_fusion.gate.grad is not None


def test_f1_checkpoint_round_trip(tmp_path):
    generator = Generator(100)
    discriminator = Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
    save_checkpoint(
        tmp_path, 0, generator, discriminator, g_optimizer, d_optimizer,
        {"g_loss": 1.0}, {"model_variant": "f1_haar_dwt"},
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
