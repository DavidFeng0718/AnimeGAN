import argparse
import platform
import sys


MIN_PYTHON = (3, 10)
MIN_TORCH = (2, 7)


def parse_version(version):
    base = version.split("+", 1)[0]
    parts = []
    for piece in base.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def collect_environment():
    import numpy as np
    import PIL
    import scipy
    import torch

    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"unavailable ({exc})"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
    }


def print_environment():
    for key, value in collect_environment().items():
        print(f"{key}: {value}")


def assert_supported(require_cuda=False):
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; found {sys.version.split()[0]}."
        )

    import torch

    torch_version = parse_version(torch.__version__)
    if torch_version < MIN_TORCH:
        raise RuntimeError(
            f"PyTorch {MIN_TORCH[0]}.{MIN_TORCH[1]}+ is required; found {torch.__version__}."
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. In Colab, set Runtime > Change runtime type > GPU.")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Print and validate the StyleGAN2-ADA runtime.")
    parser.add_argument("--check", action="store_true", help="Fail if required versions are missing.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable.")
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    print_environment()
    if parsed.check:
        assert_supported(parsed.require_cuda)


if __name__ == "__main__":
    main()
