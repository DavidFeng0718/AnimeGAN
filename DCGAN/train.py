import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

from create_dataset import My_dataset, save_img
# from model512 import Generator, Discriminator
from model64 import Generator, Discriminator


REQUIRED_CONFIG_KEYS = [
    "experiment_name",
    "image_size",
    "batch_size",
    "epochs",
    "noise_dim",
    "g_lr",
    "d_lr",
    "beta1",
    "beta2",
    "dataset_path",
    "num_workers",
    "augmentation",
]

SUPPORTED_AUGMENTATIONS = {
    "none",
    "flip",
    "brightness",
    "contrast",
    "blur",
    "sharpen",
}

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN with a JSON experiment config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file, e.g. configs/baseline.json")
    return parser.parse_args()


def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing config keys: {', '.join(missing_keys)}")

    augmentation = config["augmentation"]
    if augmentation not in SUPPORTED_AUGMENTATIONS:
        raise ValueError(
            f"Unsupported augmentation '{augmentation}'. "
            f"Choose from: {', '.join(sorted(SUPPORTED_AUGMENTATIONS))}"
        )

    return config


def create_run_dir(experiment_name):
    experiment_dir = Path("runs") / experiment_name
    while True:
        run_dir = experiment_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1)

    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "sample").mkdir(parents=True, exist_ok=True)
    (run_dir / "source_code").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_transform(config):
    transform_steps = [
        transforms.Resize((config["image_size"], config["image_size"])),
    ]

    augmentation = config["augmentation"]
    if augmentation == "flip":
        transform_steps.append(transforms.RandomHorizontalFlip(p=1.0))
    elif augmentation == "brightness":
        transform_steps.append(transforms.ColorJitter(brightness=0.2))
    elif augmentation == "contrast":
        transform_steps.append(transforms.ColorJitter(contrast=0.2))
    elif augmentation == "blur":
        transform_steps.append(transforms.GaussianBlur(kernel_size=3))
    elif augmentation == "sharpen":
        transform_steps.append(transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0))

    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(transform_steps)


def log_message(message, log_fp):
    print(message)
    log_fp.write(message + "\n")
    log_fp.flush()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def save_config(run_dir, config, dataset_size, config_path):
    config_to_save = dict(config)
    config_to_save.update({
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "dataset_size": dataset_size,
        "config_file": str(config_path),
    })
    save_json(run_dir / "config.json", config_to_save)


def save_config_used(run_dir, config_path):
    shutil.copy2(config_path, run_dir / "config_used.json")


def save_environment(run_dir, device):
    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu_name = "Apple MPS"

    environment = [
        f"Python {platform.python_version()}",
        f"Torch {torch.__version__}",
        f"CUDA {torch.version.cuda if torch.version.cuda else 'N/A'}",
        "",
        f"MPS availability: {torch.backends.mps.is_available()}",
        f"Device: {device}",
        f"GPU: {gpu_name}",
        "",
        f"OS: {platform.platform()}",
    ]
    with open(run_dir / "environment.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(environment) + "\n")


def save_source_code(run_dir):
    source_dir = run_dir / "source_code"
    source_files = ["train.py", "main.py", "model64.py", "create_dataset.py"]
    if Generator.__module__ == "model512":
        source_files.append("model512.py")

    for source_file in source_files:
        source_path = PROJECT_DIR / source_file
        if source_path.exists():
            shutil.copy2(source_path, source_dir / source_path.name)


def save_git_commit(run_dir):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "Unavailable"

    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as fp:
        fp.write(commit + "\n")


def save_model_architecture(run_dir, generator, discriminator):
    with open(run_dir / "model_architecture.txt", "w", encoding="utf-8") as fp:
        fp.write("Generator\n")
        fp.write(str(generator))
        fp.write("\n\nDiscriminator\n")
        fp.write(str(discriminator))
        fp.write("\n")


def draw_line_chart(path, x_values, series, y_label):
    width, height = 1000, 600
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 40, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    all_y_values = [value for values in series.values() for value in values]
    if not x_values or not all_y_values:
        return

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(all_y_values), max(all_y_values)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    def scale_x(value):
        return int(round(margin_left + (value - min_x) / (max_x - min_x) * plot_width))

    def scale_y(value):
        return int(round(margin_top + (max_y - value) / (max_y - min_y) * plot_height))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="black", width=2)

    for tick in range(6):
        y = margin_top + tick * plot_height / 5
        value = max_y - tick * (max_y - min_y) / 5
        draw.line((margin_left - 5, y, width - margin_right, y), fill="#e6e6e6")
        draw.text((10, y - 7), f"{value:.4f}", fill="black")

    for tick in range(6):
        x = margin_left + tick * plot_width / 5
        value = min_x + tick * (max_x - min_x) / 5
        draw.line((x, height - margin_bottom, x, height - margin_bottom + 5), fill="black")
        draw.text((x - 12, height - margin_bottom + 12), f"{value:.0f}", fill="black")

    draw.text((width // 2 - 20, height - 35), "Epoch", fill="black")
    draw.text((10, 15), y_label, fill="black")

    for series_index, (label, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = [(scale_x(x), scale_y(y)) for x, y in zip(x_values, values)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=3)
        legend_y = margin_top + series_index * 24
        legend_x = width - margin_right - 220
        draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=color, width=3)
        draw.text((legend_x + 38, legend_y), label, fill="black")

    image.save(path)


def plot_training_curves(run_dir, epoch_metrics):
    if not epoch_metrics:
        return

    epochs = [item["epoch"] for item in epoch_metrics]
    draw_line_chart(
        run_dir / "plots" / "loss_curve.png",
        epochs,
        {
            "Generator Loss": [item["avg_g_loss"] for item in epoch_metrics],
            "Discriminator Loss": [item["avg_d_loss"] for item in epoch_metrics],
        },
        "Loss",
    )
    draw_line_chart(
        run_dir / "plots" / "discriminator_curve.png",
        epochs,
        {
            "D_real": [item["avg_D_real"] for item in epoch_metrics],
            "D_fake": [item["avg_D_fake"] for item in epoch_metrics],
        },
        "Discriminator Output",
    )
    draw_line_chart(
        run_dir / "plots" / "gap_curve.png",
        epochs,
        {"D_real - D_fake": [item["avg_D_gap"] for item in epoch_metrics]},
        "D_gap",
    )


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(config, config_path):
    run_dir = create_run_dir(config["experiment_name"])
    save_config_used(run_dir, config_path)

    transform = build_transform(config)
    dataset = My_dataset(config["dataset_path"], transform=transform)   # 数据集位置
    batch_size, epochs = config["batch_size"], config["epochs"]
    my_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"], # A100情况，其余改4
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0
    )

    device = get_device()

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        betas=(config["beta1"], config["beta2"]),
        lr=config["d_lr"],
    )  # betas为adam算法两个动量参数
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        betas=(config["beta1"], config["beta2"]),
        lr=config["g_lr"],
    )
    criterion = nn.BCEWithLogitsLoss()

    save_config(run_dir, config, len(dataset), config_path)
    save_environment(run_dir, device)
    save_source_code(run_dir)
    save_git_commit(run_dir)
    save_model_architecture(run_dir, generator, discriminator)

    metrics_path = run_dir / "metrics.csv"
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    train_log_path = run_dir / "train.log"

    epoch_metrics = []
    start_time = time.time()
    final_d_loss = None
    final_g_loss = None
    best_d_loss = None
    best_g_loss = None

    with open(train_log_path, "w", encoding="utf-8") as log_fp, \
            open(metrics_path, "w", newline="", encoding="utf-8") as metrics_fp, \
            open(epoch_metrics_path, "w", newline="", encoding="utf-8") as epoch_metrics_fp:
        metrics_writer = csv.DictWriter(
            metrics_fp,
            fieldnames=["epoch", "step", "d_loss", "g_loss", "D_real", "D_fake", "D_gap"]
        )
        epoch_metrics_writer = csv.DictWriter(
            epoch_metrics_fp,
            fieldnames=["epoch", "avg_d_loss", "avg_g_loss", "avg_D_real", "avg_D_fake", "avg_D_gap"]
        )
        metrics_writer.writeheader()
        epoch_metrics_writer.writeheader()

        log_message(f"Using device: {device}", log_fp)

        for epoch in range(epochs):
            epoch_rows = []

            for i, img in enumerate(my_dataloader):

                noise = torch.randn(batch_size, config["noise_dim"], 1, 1).to(device)
                real_img = img.to(device)
                fake_img = generator(noise).detach()

                real_label = real_label = torch.full(
                    (batch_size,),
                    0.9,
                    device=device
                    )
                fake_label = torch.zeros(batch_size).to(device)
                real_out = discriminator(real_img)
                fake_out = discriminator(fake_img)
                real_loss = criterion(real_out, real_label)
                fake_loss = criterion(fake_out, fake_label)

                d_loss = real_loss + fake_loss
                d_optimizer.zero_grad()

                d_loss.backward()
                d_optimizer.step()

                noise = torch.randn(batch_size, config["noise_dim"], 1, 1).to(device)
                fake_img = generator(noise)
                output = discriminator(fake_img)

                g_loss = criterion(output, real_label)
                g_optimizer.zero_grad()

                g_loss.backward()
                g_optimizer.step()

                d_loss_value = d_loss.data.item()
                g_loss_value = g_loss.data.item()
                d_real_value = real_out.data.mean().item()
                d_fake_value = fake_out.data.mean().item()
                d_gap_value = d_real_value - d_fake_value

                row = {
                    "epoch": epoch,
                    "step": i,
                    "d_loss": d_loss_value,
                    "g_loss": g_loss_value,
                    "D_real": d_real_value,
                    "D_fake": d_fake_value,
                    "D_gap": d_gap_value,
                }
                metrics_writer.writerow(row)
                metrics_fp.flush()
                epoch_rows.append(row)

                final_d_loss = d_loss_value
                final_g_loss = g_loss_value
                best_d_loss = d_loss_value if best_d_loss is None else min(best_d_loss, d_loss_value)
                best_g_loss = g_loss_value if best_g_loss is None else min(best_g_loss, g_loss_value)

                if (i + 1) % 5 == 0:
                    log_message('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                                'D_real: {:.6f},D_fake: {:.6f}'.format(
                        epoch, epochs, d_loss_value, g_loss_value,
                        d_real_value, d_fake_value  # 打印的是真实图片的损失均值
                    ), log_fp)
                if epoch == 0 and i == len(my_dataloader) - 1:          # 保存真实图像
                    save_img(img[:64, :, :, :], run_dir / "sample" / "real_images.png")
                if (epoch+1) % 10 == 0 and i == len(my_dataloader)-1:             # 每10个epoch保存一次预测图像
                    save_img(fake_img[:64, :, :, :], run_dir / "sample" / "fake_images_{}.png".format(epoch + 1))

            if epoch_rows:
                avg_d_loss = sum(row["d_loss"] for row in epoch_rows) / len(epoch_rows)
                avg_g_loss = sum(row["g_loss"] for row in epoch_rows) / len(epoch_rows)
                avg_D_real = sum(row["D_real"] for row in epoch_rows) / len(epoch_rows)
                avg_D_fake = sum(row["D_fake"] for row in epoch_rows) / len(epoch_rows)
                avg_D_gap = sum(row["D_gap"] for row in epoch_rows) / len(epoch_rows)
                epoch_row = {
                    "epoch": epoch,
                    "avg_d_loss": avg_d_loss,
                    "avg_g_loss": avg_g_loss,
                    "avg_D_real": avg_D_real,
                    "avg_D_fake": avg_D_fake,
                    "avg_D_gap": avg_D_gap,
                }
                epoch_metrics_writer.writerow(epoch_row)
                epoch_metrics_fp.flush()
                epoch_metrics.append(epoch_row)

    checkpoint_dir = run_dir / "checkpoints"
    torch.save(generator.state_dict(), checkpoint_dir / "generator.pth")        # 保存权重文件
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator.pth")
    torch.save(generator.state_dict(), checkpoint_dir / "generator_final.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_final.pth")

    plot_training_curves(run_dir, epoch_metrics)

    summary = {
        "experiment_name": config["experiment_name"],
        "augmentation": config["augmentation"],
        "config_file": str(config_path),
        "epochs": epochs,
        "dataset_size": len(dataset),
        "training_time_hours": (time.time() - start_time) / 3600,
        "final_d_loss": final_d_loss,
        "final_g_loss": final_g_loss,
        "best_d_loss": best_d_loss,
        "best_g_loss": best_g_loss,
    }
    save_json(run_dir / "summary.json", summary)
    return run_dir


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    train(config, config_path)


if __name__ == "__main__":
    main()
