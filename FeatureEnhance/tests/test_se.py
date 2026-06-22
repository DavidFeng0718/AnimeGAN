import pytest
import torch

from feature_modules import SEBlock
from model64 import Generator as BaseGenerator
from model64 import Discriminator as BaseDiscriminator
from model64_se import Generator
from train import init_dcgan_weights


DEVICES = [torch.device("cpu")]
if torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


def test_se_hidden_width_has_lower_bound():
    block = SEBlock(channels=8, reduction=16)
    assert block.hidden_channels == 1
    with pytest.raises(ValueError):
        SEBlock(channels=8, reduction=0)


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device.type)
def test_a1_shape_dtype_device_and_gradients(device):
    generator = Generator(noise_dim=100, se_reduction=16).to(device)
    discriminator = BaseDiscriminator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    noise = torch.randn(2, 100, 1, 1, device=device, requires_grad=True)
    generated = generator(noise)
    assert generated.shape == (2, 3, 64, 64)
    assert generated.dtype == noise.dtype
    assert generated.device == noise.device
    scores = discriminator(generated)
    optimizer.zero_grad(set_to_none=True)
    scores.mean().backward()
    assert noise.grad is not None
    se_parameters = [
        parameter
        for name, parameter in generator.named_parameters()
        if ".excitation." in name
    ]
    assert se_parameters
    assert all(parameter.grad is not None for parameter in se_parameters)
    optimizer.step()


def test_a1_is_a_small_generator_only_change():
    generator = Generator()
    discriminator = BaseDiscriminator()
    generator_count = sum(parameter.numel() for parameter in generator.parameters())
    discriminator_count = sum(parameter.numel() for parameter in discriminator.parameters())
    assert generator_count == 1101449
    assert discriminator_count == 1437505
    assert generator_count - 1100707 == 742
    blocks = [module for module in generator.modules() if isinstance(module, SEBlock)]
    assert [block.channels for block in blocks] == [64, 32]


def test_a1_preserves_shared_base_initialization():
    torch.manual_seed(42)
    base = BaseGenerator(100)
    base.apply(init_dcgan_weights)

    torch.manual_seed(42)
    attention = Generator(100)
    attention.apply(init_dcgan_weights)
    shared_parameters = (
        list(attention.block_4.parameters())
        + list(attention.block_8.parameters())
        + list(attention.block_16.parameters())
        + list(attention.block_32.parameters())
        + list(attention.to_rgb.parameters())
    )
    for base_parameter, attention_parameter in zip(base.parameters(), shared_parameters):
        assert torch.equal(base_parameter, attention_parameter)
