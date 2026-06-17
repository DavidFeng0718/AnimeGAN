import argparse
import heapq
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from model64 import Generator as Generator64, Discriminator as Discriminator64
from model512 import Generator as Generator512, Discriminator as Discriminator512


MODEL_REGISTRY = {
    64: (Generator64, Discriminator64),
    512: (Generator512, Discriminator512),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a trained DCGAN checkpoint.")
    parser.add_argument("--config", default="configs/baseline.json", help="Training config used by the checkpoint.")
    parser.add_argument("--generator", default="generator.pth", help="Generator checkpoint path.")
    parser.add_argument("--discriminator", default="discriminator.pth", help="Discriminator checkpoint path.")
    parser.add_argument("--save-dir", default="genImage", help="Directory for generated images.")
    parser.add_argument("--candidate-num", type=int, default=10000, help="Number of candidates to generate.")
    parser.add_argument("--save-num", type=int, default=100, help="Number of top candidates to save.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    image_size = config.get("image_size")
    if image_size not in MODEL_REGISTRY:
        supported_sizes = ", ".join(str(size) for size in sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported image_size {image_size}. Supported sizes: {supported_sizes}")

    return config


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


def main():
    args = parse_args()
    if args.candidate_num <= 0 or args.save_num <= 0:
        raise ValueError("candidate-num and save-num must be positive.")
    if args.save_num > args.candidate_num:
        raise ValueError("save-num cannot be larger than candidate-num.")

    config = load_config(args.config)
    device = get_device()
    print("Using device:", device)

    Generator, Discriminator = MODEL_REGISTRY[config["image_size"]]
    generator = Generator(noise_dim=config["noise_dim"]).to(device)
    discriminator = Discriminator().to(device)

    generator.load_state_dict(
        torch.load(args.generator, map_location=device)
    )
    discriminator.load_state_dict(
        torch.load(args.discriminator, map_location=device)
    )

    generator.eval()
    discriminator.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
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

            if len(candidates) < args.save_num:
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
