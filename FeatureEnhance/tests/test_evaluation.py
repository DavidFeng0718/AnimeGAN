import numpy as np
import torch
from PIL import Image

from evaluate_features import (
    EvaluationImageDataset,
    choose_inception_device,
    compute_basic_metrics,
    deterministic_pairs,
    image_border_statistics,
    image_laplacian_variance,
)


def test_inception_metrics_fall_back_to_cpu_for_mps():
    assert choose_inception_device(torch.device("mps")) == torch.device("cpu")
    assert choose_inception_device(torch.device("cpu")) == torch.device("cpu")
    assert choose_inception_device(torch.device("cuda")) == torch.device("cuda")


def test_evaluation_io_and_diversity_smoke(tmp_path):
    for index, value in enumerate((0, 64, 128, 255)):
        pixels = np.full((64, 64, 3), value, dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / f"sample_{index}.png")
    dataset = EvaluationImageDataset(tmp_path, image_size=64)
    config = {
        "evaluation_batch_size": 2,
        "lpips_pairs": 4,
        "generation_seed": 42,
        "lpips_batch_size": 2,
        "duplicate_mse_threshold": 0.0001,
    }
    result = compute_basic_metrics(dataset, config)
    assert result["sampled_pair_count"] == 4
    assert result["exact_duplicate_rate"] == 0.0
    assert result["sampled_pair_mse_mean"] > 0
    assert "border_luminance_gap_mean" in result
    assert all(left != right for left, right in deterministic_pairs(4, 20, 42))


def test_border_metric_detects_dark_frame():
    images = torch.ones(1, 3, 64, 64)
    images[:, :, :4, :] = 0
    images[:, :, -4:, :] = 0
    images[:, :, :, :4] = 0
    images[:, :, :, -4:] = 0
    metrics = image_border_statistics(images)
    assert metrics["border_luminance_gap"].item() == -1.0
    assert metrics["border_dark_pixel_ratio"].item() == 1.0


def test_valid_laplacian_does_not_count_zero_padding_as_detail():
    images = torch.ones(1, 3, 64, 64)
    assert image_laplacian_variance(images, padding=0).item() == 0.0
    assert image_laplacian_variance(images, padding=1).item() > 0.0
