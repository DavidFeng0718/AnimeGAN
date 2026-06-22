import argparse
import heapq
import json
import random
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from model64 import Generator as Generator64, Discriminator as Discriminator64
from model64_se import Generator as Generator64SE
from model64_sn import Discriminator as Discriminator64SN
from model64_sobel import Generator as Generator64Sobel
from model64_dwt import Generator as Generator64DWT
from model512 import Generator as Generator512, Discriminator as Discriminator512


MODEL_REGISTRY = {
    64: (Generator64, Discriminator64),
    512: (Generator512, Discriminator512),
}
MODEL_VARIANT_REGISTRY = {
    "base": (Generator64, Discriminator64, 64),
    "a1_se": (Generator64SE, Discriminator64, 64),
    "s1_spectralnorm": (Generator64, Discriminator64SN, 64),
    "e1_sobel": (Generator64Sobel, Discriminator64, 64),
    "f1_haar_dwt": (Generator64DWT, Discriminator64, 64),
}
DEFAULT_CONFIG_PATH = "configs/FeatureEnhance/d1_diffaugment.json"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Generate images from a trained DCGAN checkpoint.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Training config used by the checkpoint (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--generator", required=True, help="Generator checkpoint path.")
    parser.add_argument("--discriminator", default="discriminator.pth", help="Discriminator checkpoint path.")
    parser.add_argument("--save-dir", default="genImage", help="Directory for generated images.")
    parser.add_argument("--candidate-num", type=int, default=10000, help="Number of candidates to generate.")
    parser.add_argument("--save-num", type=int, default=None, help="Number of images to save.")
    parser.add_argument(
        "--mode", choices=("ranked", "independent"), default="ranked",
        help="Keep legacy discriminator ranking or save every independent sample.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Generation batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Generation seed.")
    return parser.parse_args(args)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    image_size = config.get("image_size")
    if image_size not in MODEL_REGISTRY:
        supported_sizes = ", ".join(str(size) for size in sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported image_size {image_size}. Supported sizes: {supported_sizes}")

    return config


def get_model_classes(config):
    model_variant = config.get("model_variant")
    if model_variant is None:
        return MODEL_REGISTRY[config["image_size"]]
    if model_variant not in MODEL_VARIANT_REGISTRY:
        raise ValueError(f"Unsupported model_variant: {model_variant}")
    Generator, Discriminator, image_size = MODEL_VARIANT_REGISTRY[model_variant]
    if config["image_size"] != image_size:
        raise ValueError(f"model_variant {model_variant} requires image_size={image_size}.")
    return Generator, Discriminator


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_start_id(save_dir):
    existing_ids = []
    for path in save_dir.glob("generated_*.png"):
        try:
            existing_ids.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(existing_ids, default=0) + 1


def load_generator_state(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        return checkpoint["generator_state_dict"]
    return checkpoint


def generate_independent(generator, save_dir, config, count, batch_size, device):
    if count < 4545:
        raise ValueError("P0 independent evaluation requires at least 4545 samples.")
    if any(save_dir.glob("generated_*.png")):
        raise FileExistsError(f"Independent output must be empty: {save_dir}")
    started = time.perf_counter()
    saved = 0
    with torch.inference_mode():
        while saved < count:
            current_batch = min(batch_size, count - saved)
            noise = torch.randn(current_batch, config["noise_dim"], 1, 1, device=device)
            images = generator(noise).cpu()
            for offset, image in enumerate(images):
                save_image(
                    image,
                    save_dir / f"generated_{saved + offset + 1:06d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
            saved += current_batch
            print(f"Generated {saved}/{count}", flush=True)
    return time.perf_counter() - started


def main():
    args = parse_args()
    config = load_config(args.config)
    save_num = args.save_num or (
        config.get("generation_count", 4545) if args.mode == "independent" else 100
    )
    batch_size = args.batch_size or config.get("generation_batch_size", 128)
    seed = args.seed if args.seed is not None else config.get("generation_seed", 42)
    if args.candidate_num <= 0 or save_num <= 0 or batch_size <= 0:
        raise ValueError("candidate-num and save-num must be positive.")
    if save_num > args.candidate_num and args.mode == "ranked":
        raise ValueError("save-num cannot be larger than candidate-num.")

    random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    print("Using device:", device)

    Generator, Discriminator = get_model_classes(config)
    generator = Generator(noise_dim=config["noise_dim"]).to(device)
    generator.load_state_dict(load_generator_state(args.generator, device))

    generator.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "independent":
        seconds = generate_independent(
            generator, save_dir, config, save_num, batch_size, device
        )
        manifest = {
            "mode": "independent",
            "checkpoint": str(Path(args.generator).resolve()),
            "sample_count": save_num,
            "batch_size": batch_size,
            "seed": seed,
            "seconds": seconds,
        }
        with open(save_dir / "manifest.json", "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)
            fp.write("\n")
        print(f"Saved {save_num} independent images to {save_dir}")
        return

    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load(args.discriminator, map_location=device))
    discriminator.eval()
    start_id = get_start_id(save_dir)

    candidates = []
    with torch.no_grad():
        for i in range(args.candidate_num):
            noise = torch.randn(
                1,
                config["noise_dim"],
                1,
                1,
                device=device
            )
            img = generator(noise)
            score = discriminator(img).item()
            entry = (score, i, img.cpu())

            if len(candidates) < save_num:
                heapq.heappush(candidates, entry)
            elif score > candidates[0][0]:
                heapq.heapreplace(candidates, entry)

            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{args.candidate_num}")

    print("Saving top candidates...")
    candidates.sort(key=lambda item: item[0], reverse=True)
    for i, (score, _, img) in enumerate(candidates):
        img = (img + 1) / 2
        save_path = save_dir / f"generated_{start_id + i:04d}.png"
        save_image(img, save_path)
        print(f"Saved {save_path} | Score={score:.4f}")

    print("\nFinished!")
    print(f"Top {len(candidates)} images saved.")


if __name__ == "__main__":
    main()
