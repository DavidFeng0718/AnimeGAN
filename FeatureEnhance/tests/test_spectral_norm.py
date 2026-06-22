import torch

from model64 import Discriminator as BaseDiscriminator, Generator
from model64_sn import Discriminator as SpectralNormDiscriminator
from train import init_dcgan_weights, restore_training_checkpoint, save_checkpoint


def test_spectral_norm_discriminator_shape_gradients_and_parameters():
    discriminator = SpectralNormDiscriminator()
    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    scores = discriminator(images)
    assert scores.shape == (2,)
    scores.mean().backward()
    assert images.grad is not None
    assert all(parameter.grad is not None for parameter in discriminator.parameters())
    assert sum(parameter.numel() for parameter in discriminator.parameters()) == 1437505

    weighted_layers = [
        module for module in discriminator.modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    assert len(weighted_layers) == 6
    assert all(hasattr(module, "weight_orig") for module in weighted_layers)
    assert all(hasattr(module, "weight_u") for module in weighted_layers)


def test_spectral_norm_preserves_paired_generator_initialization():
    torch.manual_seed(42)
    base_discriminator = BaseDiscriminator()
    base_generator = Generator(100)
    base_discriminator.apply(init_dcgan_weights)
    base_generator.apply(init_dcgan_weights)

    torch.manual_seed(42)
    sn_discriminator = SpectralNormDiscriminator()
    sn_generator = Generator(100)
    sn_discriminator.apply(init_dcgan_weights)
    sn_generator.apply(init_dcgan_weights)

    for base_parameter, sn_parameter in zip(base_generator.parameters(), sn_generator.parameters()):
        assert torch.equal(base_parameter, sn_parameter)


def test_spectral_norm_checkpoint_round_trip(tmp_path):
    generator = Generator(100)
    discriminator = SpectralNormDiscriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
    discriminator(torch.randn(2, 3, 64, 64))
    save_checkpoint(
        tmp_path, 0, generator, discriminator, g_optimizer, d_optimizer,
        {"g_loss": 1.0}, {"model_variant": "s1_spectralnorm"},
    )

    restored_generator = Generator(100)
    restored_discriminator = SpectralNormDiscriminator()
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
    for expected, actual in zip(discriminator.state_dict().values(), restored_discriminator.state_dict().values()):
        assert torch.equal(expected, actual)
