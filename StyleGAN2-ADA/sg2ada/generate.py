import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from .paths import UPSTREAM_DIR

sys.path.insert(0, str(UPSTREAM_DIR))

import dnnlib  # noqa: E402
import legacy  # noqa: E402


def parse_seed_list(value):
    if value is None:
        return None
    seeds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate PNG samples from a StyleGAN2-ADA network pickle."
    )
    parser.add_argument("--network", required=True, help="Path or URL to network-snapshot-*.pkl.")
    parser.add_argument("--outdir", required=True, help="Directory for generated images.")
    parser.add_argument("--count", type=int, default=4545, help="Number of images to generate.")
    parser.add_argument("--seeds", help="Comma/range seed list, e.g. 0-63,100,102.")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation batch size.")
    parser.add_argument("--start-seed", type=int, default=0, help="First random seed.")
    parser.add_argument("--trunc", type=float, default=1.0, help="Truncation psi.")
    parser.add_argument(
        "--noise-mode",
        choices=("const", "random", "none"),
        default="const",
        help="Noise mode passed to the generator.",
    )
    parser.add_argument("--class", dest="class_idx", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Generation device.",
    )
    parser.add_argument("--projected-w", help="NPZ file containing projected W vectors.")
    parser.add_argument(
        "--interpolate",
        nargs=2,
        type=int,
        metavar=("SEED_A", "SEED_B"),
        help="Generate a latent interpolation between two seeds.",
    )
    parser.add_argument("--interpolation-steps", type=int, default=16)
    parser.add_argument("--grid", action="store_true", help="Also write grid.png.")
    parser.add_argument("--grid-width", type=int, default=0, help="Grid columns; default is sqrt(N).")
    parser.add_argument("--grid-only", action="store_true", help="Write only grid.png.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting generated images that already exist.",
    )
    return parser.parse_args(args)


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but CUDA is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but Apple MPS is not available.")
    return torch.device(requested)


def network_source(path_or_url):
    path = Path(path_or_url).expanduser()
    if path.exists():
        return str(path.resolve())
    return path_or_url


def build_label(generator, class_idx, batch_size, device):
    label = torch.zeros([batch_size, generator.c_dim], device=device)
    if generator.c_dim == 0:
        if class_idx is not None:
            print("warning: --class is ignored for an unconditional network")
        return label
    if class_idx is None:
        raise ValueError("Conditional network requires --class.")
    if class_idx < 0 or class_idx >= generator.c_dim:
        raise ValueError(f"--class must be in [0, {generator.c_dim - 1}].")
    label[:, class_idx] = 1
    return label


def seeds_to_z(seeds, z_dim, device):
    z = np.stack([
        np.random.RandomState(seed).randn(z_dim).astype(np.float32)
        for seed in seeds
    ])
    return torch.from_numpy(z).to(device)


def z_for_seed(seed, z_dim, device):
    return seeds_to_z([seed], z_dim, device)


def save_image(tensor, path):
    image = (tensor.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(image.cpu().numpy(), "RGB").save(path)


def tensor_to_pil(tensor):
    image = (tensor.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(image.cpu().numpy(), "RGB")


def save_grid(images, path, width=0):
    if not images:
        return
    if width <= 0:
        width = math.ceil(math.sqrt(len(images)))
    height = math.ceil(len(images) / width)
    cell_w, cell_h = images[0].size
    grid = PIL.Image.new("RGB", (width * cell_w, height * cell_h), "black")
    for idx, image in enumerate(images):
        x = (idx % width) * cell_w
        y = (idx // width) * cell_h
        grid.paste(image, (x, y))
    grid.save(path)


def write_manifest(path, args, device, seconds, seeds, mode):
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(Path(args.network).expanduser()),
        "outdir": str(Path(args.outdir).resolve()),
        "mode": mode,
        "count": len(seeds),
        "batch_size": args.batch_size,
        "seeds": seeds,
        "truncation_psi": args.trunc,
        "noise_mode": args.noise_mode,
        "class_idx": args.class_idx,
        "device": str(device),
        "seconds": seconds,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_output_dir(outdir, overwrite):
    outdir.mkdir(parents=True, exist_ok=True)
    existing = list(outdir.glob("generated_*.png")) + list(outdir.glob("seed_*.png"))
    existing += list(outdir.glob("interp_*.png"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{outdir} already contains generated outputs. Use --overwrite or choose an empty directory."
        )


def load_generator(network, device):
    print(f'Loading network from "{network}" on {device}...')
    with dnnlib.util.open_url(network_source(network)) as handle:
        return legacy.load_network_pkl(handle)["G_ema"].to(device).eval()


def generate_projected_w(generator, args, device, outdir):
    ws = np.load(args.projected_w)["w"]
    ws = torch.tensor(ws, device=device)
    if ws.shape[1:] != (generator.num_ws, generator.w_dim):
        raise ValueError(f"Projected W shape must end with {(generator.num_ws, generator.w_dim)}.")
    images = []
    with torch.inference_mode():
        for idx, w in enumerate(ws):
            image = generator.synthesis(w.unsqueeze(0), noise_mode=args.noise_mode)[0]
            pil_image = tensor_to_pil(image)
            images.append(pil_image)
            if not args.grid_only:
                pil_image.save(outdir / f"projected_{idx:04d}.png")
    return images, list(range(len(images))), "projected_w"


def generate_interpolation(generator, args, device, outdir):
    if args.interpolation_steps <= 1:
        raise ValueError("--interpolation-steps must be greater than 1.")
    seed_a, seed_b = args.interpolate
    z_a = z_for_seed(seed_a, generator.z_dim, device)
    z_b = z_for_seed(seed_b, generator.z_dim, device)
    label = build_label(generator, args.class_idx, 1, device)
    images = []
    seeds = [seed_a, seed_b]
    with torch.inference_mode():
        for idx, alpha in enumerate(np.linspace(0.0, 1.0, args.interpolation_steps)):
            z = (1.0 - float(alpha)) * z_a + float(alpha) * z_b
            image = generator(z, label, truncation_psi=args.trunc, noise_mode=args.noise_mode)[0]
            pil_image = tensor_to_pil(image)
            images.append(pil_image)
            if not args.grid_only:
                pil_image.save(outdir / f"interp_{idx:04d}.png")
    return images, seeds, "interpolation"


def generate_seed_images(generator, args, device, outdir):
    seeds = parse_seed_list(args.seeds)
    explicit_seeds = seeds is not None
    if seeds is None:
        seeds = list(range(args.start_seed, args.start_seed + args.count))
    if not seeds:
        raise ValueError("No seeds requested.")
    if args.count <= 0 or args.batch_size <= 0:
        raise ValueError("--count and --batch-size must be positive.")

    images = []
    saved = 0
    with torch.inference_mode():
        while saved < len(seeds):
            batch_seeds = seeds[saved : saved + args.batch_size]
            current_batch = len(batch_seeds)
            z = seeds_to_z(batch_seeds, generator.z_dim, device)
            label = build_label(generator, args.class_idx, current_batch, device)
            batch_images = generator(
                z,
                label,
                truncation_psi=args.trunc,
                noise_mode=args.noise_mode,
            )
            for offset, image in enumerate(batch_images):
                seed = batch_seeds[offset]
                pil_image = tensor_to_pil(image)
                images.append(pil_image)
                if not args.grid_only:
                    if explicit_seeds:
                        path = outdir / f"seed_{seed:06d}.png"
                    else:
                        path = outdir / f"generated_{saved + offset + 1:06d}.png"
                    pil_image.save(path)
            saved += current_batch
            print(f"Generated {saved}/{len(seeds)}", flush=True)
    return images, seeds, "seeds"


def main(args=None):
    parsed = parse_args(args)
    if parsed.device in {"auto", "mps"}:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = choose_device(parsed.device)
    outdir = Path(parsed.outdir)
    validate_output_dir(outdir, parsed.overwrite)
    generator = load_generator(parsed.network, device)

    started = time.perf_counter()
    if parsed.projected_w:
        images, seeds, mode = generate_projected_w(generator, parsed, device, outdir)
    elif parsed.interpolate:
        images, seeds, mode = generate_interpolation(generator, parsed, device, outdir)
    else:
        images, seeds, mode = generate_seed_images(generator, parsed, device, outdir)

    if parsed.grid or parsed.grid_only:
        save_grid(images, outdir / "grid.png", parsed.grid_width)
    seconds = time.perf_counter() - started
    write_manifest(outdir / "manifest.json", parsed, device, seconds, seeds, mode)
    print(f"Saved {len(images)} images to {outdir.resolve()}")
