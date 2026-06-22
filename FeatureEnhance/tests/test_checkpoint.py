import random

import pytest
import torch

from model64 import Discriminator, Generator
from model64_se import Generator as SEGenerator
from train import restore_training_checkpoint, save_checkpoint


def test_full_checkpoint_round_trip(tmp_path):
    generator, discriminator = Generator(100), Discriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
    save_checkpoint(
        tmp_path, 0, generator, discriminator, generator_optimizer,
        discriminator_optimizer, {"g_loss": 1.0}, {"model_variant": "base"},
    )
    checkpoint = tmp_path / "checkpoint_epoch_1.pt"
    restored_generator, restored_discriminator = Generator(100), Discriminator()
    restored_g_optimizer = torch.optim.Adam(restored_generator.parameters(), lr=0.0002)
    restored_d_optimizer = torch.optim.Adam(restored_discriminator.parameters(), lr=0.00005)
    start_epoch = restore_training_checkpoint(
        checkpoint, restored_generator, restored_discriminator,
        restored_g_optimizer, restored_d_optimizer, torch.device("cpu"),
    )
    assert start_epoch == 1
    for expected, actual in zip(generator.parameters(), restored_generator.parameters()):
        assert torch.equal(expected, actual)


def test_a1_checkpoint_round_trip(tmp_path):
    generator, discriminator = SEGenerator(100), Discriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
    save_checkpoint(
        tmp_path, 0, generator, discriminator, generator_optimizer,
        discriminator_optimizer, {"g_loss": 1.0}, {"model_variant": "a1_se"},
    )
    restored_generator, restored_discriminator = SEGenerator(100), Discriminator()
    restored_g_optimizer = torch.optim.Adam(restored_generator.parameters(), lr=0.0002)
    restored_d_optimizer = torch.optim.Adam(restored_discriminator.parameters(), lr=0.00005)
    start_epoch = restore_training_checkpoint(
        tmp_path / "checkpoint_epoch_1.pt", restored_generator,
        restored_discriminator, restored_g_optimizer, restored_d_optimizer,
        torch.device("cpu"),
    )
    assert start_epoch == 1
    for expected, actual in zip(generator.parameters(), restored_generator.parameters()):
        assert torch.equal(expected, actual)


def test_checkpoint_restores_rng_state_and_rejects_config_mismatch(tmp_path):
    random.seed(123)
    torch.manual_seed(123)
    generator, discriminator = Generator(100), Discriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
    save_checkpoint(
        tmp_path, 0, generator, discriminator, generator_optimizer,
        discriminator_optimizer, {"g_loss": 1.0}, {"model_variant": "base"},
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)
    random.seed(999)
    torch.manual_seed(999)

    restore_training_checkpoint(
        tmp_path / "checkpoint_epoch_1.pt", generator, discriminator,
        generator_optimizer, discriminator_optimizer, torch.device("cpu"),
    )
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)

    with pytest.raises(ValueError, match="model_variant"):
        restore_training_checkpoint(
            tmp_path / "checkpoint_epoch_1.pt", generator, discriminator,
            generator_optimizer, discriminator_optimizer, torch.device("cpu"),
            current_config={"model_variant": "a1_se"},
        )
