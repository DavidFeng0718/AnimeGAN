from GAN_generator import (
    DEFAULT_CONFIG_PATH as GENERATOR_DEFAULT_CONFIG_PATH,
    parse_args as parse_generator_args,
)
from train import DEFAULT_CONFIG_PATH, load_config, parse_args as parse_train_args


def test_training_and_generation_default_to_passed_d1_config():
    expected = "configs/FeatureEnhance/d1_diffaugment.json"
    assert DEFAULT_CONFIG_PATH == expected
    assert GENERATOR_DEFAULT_CONFIG_PATH == expected
    assert parse_train_args([]).config == expected
    assert parse_generator_args(["--generator", "checkpoint.pt"]).config == expected
    config = load_config(expected)
    assert config["experiment_name"] == "featureEnhance/d1_diffaugment_v1"
    assert config["diff_augment_policy"] == "color,translation,cutout"
