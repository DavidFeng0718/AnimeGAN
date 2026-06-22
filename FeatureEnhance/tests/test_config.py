import copy
from pathlib import Path

import pytest

from create_dataset import IMAGE_EXTS
from train import (
    CURRENT_PREPROCESSING_VERSION,
    FEATURE_BASE_REQUIRED_KEYS,
    get_model_classes,
    load_config,
    resolve_project_path,
    validate_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance" / "base.json"
A1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance" / "a1_se.json"
D1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance" / "d1_diffaugment.json"
S1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance" / "s1_spectralnorm.json"
E1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance" / "e1_sobel.json"
F1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance" / "f1_haar_dwt.json"
FEATURE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "FeatureEnhance"


def test_base_config_is_explicit_and_points_to_4545_images():
    config = load_config(CONFIG_PATH)
    assert FEATURE_BASE_REQUIRED_KEYS <= config.keys()
    assert config["experiment_name"] == "featureEnhance/base_v3"
    assert config["model_variant"] == "base"
    assert config["schema_version"] == 3
    assert config["preprocessing_version"] == CURRENT_PREPROCESSING_VERSION
    assert config["rotation_probability"] == 0.6
    assert config["rotation_degrees"] == 6.0
    assert config["epochs"] == 300
    assert config["seed"] == 42
    assert config["augmentation"] == "rotation+color_jitter+flip"
    assert config["diff_augment_policy"] == "none"
    Generator, Discriminator, model_file = get_model_classes(config)
    assert Generator.__module__ == "model64"
    assert Discriminator.__module__ == "model64"
    assert model_file == "model64.py"
    images = [
        path for path in resolve_project_path(config["dataset_path"]).iterdir()
        if path.suffix.lower() in IMAGE_EXTS
    ]
    assert len(images) == 4545


def test_config_allows_zero_workers_and_rejects_invalid_training_values():
    config = load_config(CONFIG_PATH)
    zero_workers = copy.deepcopy(config)
    zero_workers["num_workers"] = 0
    validate_config(zero_workers)

    for key, value in (("g_lr", 0), ("d_lr", -0.1), ("image_channels", 1)):
        invalid = copy.deepcopy(config)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_feature_config_rejects_unversioned_preprocessing():
    config = load_config(CONFIG_PATH)
    config["schema_version"] = 2
    config.pop("preprocessing_version")
    with pytest.raises(ValueError):
        validate_config(config)


def test_diffaugment_config_is_a_strict_single_variable_experiment():
    base = load_config(CONFIG_PATH)
    diffaugment = load_config(D1_CONFIG_PATH)
    changed = {
        key for key in base
        if base[key] != diffaugment[key]
    }
    assert changed == {"experiment_name", "diff_augment_policy"}
    assert diffaugment["model_variant"] == "base"


def test_se_config_is_a_strict_single_variable_experiment():
    base = load_config(CONFIG_PATH)
    attention = load_config(A1_CONFIG_PATH)
    changed = {key for key in base if base[key] != attention[key]}
    assert changed == {"experiment_name", "model_variant"}
    Generator, Discriminator, model_file = get_model_classes(attention)
    assert Generator.__module__ == "model64_se"
    assert Discriminator.__module__ == "model64"
    assert model_file == "model64_se.py"


def test_diffaugment_config_rejects_invalid_policy_and_ratios():
    config = load_config(D1_CONFIG_PATH)
    for key, value in (
        ("diff_augment_policy", "color,unknown"),
        ("diff_augment_translation_ratio", 0.75),
        ("diff_augment_cutout_ratio", -0.1),
    ):
        invalid = copy.deepcopy(config)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_spectralnorm_config_is_a_strict_single_variable_experiment():
    base = load_config(CONFIG_PATH)
    spectralnorm = load_config(S1_CONFIG_PATH)
    changed = {key for key in base if base[key] != spectralnorm[key]}
    assert changed == {"experiment_name", "model_variant"}
    Generator, Discriminator, model_file = get_model_classes(spectralnorm)
    assert Generator.__module__ == "model64"
    assert Discriminator.__module__ == "model64_sn"
    assert model_file == "model64_sn.py"


def test_sobel_config_is_a_strict_single_variable_experiment():
    base = load_config(CONFIG_PATH)
    sobel = load_config(E1_CONFIG_PATH)
    changed = {key for key in base if base[key] != sobel[key]}
    assert changed == {"experiment_name", "model_variant"}
    Generator, Discriminator, model_file = get_model_classes(sobel)
    assert Generator.__module__ == "model64_sobel"
    assert Discriminator.__module__ == "model64"
    assert model_file == "model64_sobel.py"


def test_haar_dwt_config_is_a_strict_single_variable_experiment():
    base = load_config(CONFIG_PATH)
    dwt = load_config(F1_CONFIG_PATH)
    changed = {key for key in base if base[key] != dwt[key]}
    assert changed == {"experiment_name", "model_variant"}
    Generator, Discriminator, model_file = get_model_classes(dwt)
    assert Generator.__module__ == "model64_dwt"
    assert Discriminator.__module__ == "model64"
    assert model_file == "model64_dwt.py"


@pytest.mark.parametrize("seed", [123, 3407])
def test_multiseed_base_and_diffaugment_configs_are_paired(seed):
    base = load_config(FEATURE_CONFIG_DIR / f"base_seed{seed}.json")
    diffaugment = load_config(FEATURE_CONFIG_DIR / f"d1_diffaugment_seed{seed}.json")
    changed = {key for key in base if base[key] != diffaugment[key]}
    assert changed == {"experiment_name", "diff_augment_policy"}
    assert base["seed"] == diffaugment["seed"] == seed
    assert base["fixed_noise_seed"] == diffaugment["fixed_noise_seed"] == 10000 + seed
    assert base["generation_seed"] == diffaugment["generation_seed"] == 10000 + seed


@pytest.mark.parametrize("seed", [123, 3407])
def test_multiseed_base_and_se_configs_are_paired(seed):
    base = load_config(FEATURE_CONFIG_DIR / f"base_seed{seed}.json")
    attention = load_config(FEATURE_CONFIG_DIR / f"a1_se_seed{seed}.json")
    changed = {key for key in base if base[key] != attention[key]}
    assert changed == {"experiment_name", "model_variant"}
    assert base["seed"] == attention["seed"] == seed
    assert base["fixed_noise_seed"] == attention["fixed_noise_seed"] == 10000 + seed
    assert base["generation_seed"] == attention["generation_seed"] == 10000 + seed
