import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "featureEnhance" / "feature_comparison"
FIXED_EXPERIMENTS = {
    "Base v3": (
        "runs/featureEnhance/base_v3/20260619_180759",
        200,
    ),
    "Sobel": (
        "runs/featureEnhance/e1_sobel_v1/20260620_072940",
        100,
    ),
    "Haar DWT": (
        "runs/featureEnhance/f1_haar_dwt_v1/20260620_085338",
        100,
    ),
    "D1 DiffAugment": (
        "runs/featureEnhance/d1_diffaugment_v1/20260619_205038",
        200,
    ),
}
METRICS = (
    ("KID", "kid_mean", False),
    ("FID", "fid", False),
    ("LPIPS diversity", "lpips_diversity_mean", True),
    ("Valid Laplacian", "laplacian_variance_valid_mean", True),
)
COLORS = {
    "Base v3": "#777777",
    "SE attention": "#3b82f6",
    "Sobel": "#ef4444",
    "Haar DWT": "#f59e0b",
    "D1 DiffAugment": "#10b981",
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Create a same-protocol feature-module metric and sample report."
    )
    parser.add_argument("--attention-run", required=True, help="Completed A1 SE v3 run directory.")
    parser.add_argument("--attention-epoch", required=True, type=int, help="Selected A1 epoch.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--se-multiseed-summary",
        default=None,
        help="Optional SE multiseed summary JSON used to draw paired seed bars.",
    )
    return parser.parse_args(args)


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent_change(value, baseline):
    return (value / baseline - 1.0) * 100.0


def load_experiments(attention_run, attention_epoch):
    manifest = {
        "Base v3": FIXED_EXPERIMENTS["Base v3"],
        "SE attention": (str(resolve(attention_run)), attention_epoch),
        "Sobel": FIXED_EXPERIMENTS["Sobel"],
        "Haar DWT": FIXED_EXPERIMENTS["Haar DWT"],
        "D1 DiffAugment": FIXED_EXPERIMENTS["D1 DiffAugment"],
    }
    experiments = {}
    for name, (run_path, epoch) in manifest.items():
        run_dir = resolve(run_path)
        evaluation_path = run_dir / "evaluation" / f"epoch_{epoch}.json"
        sample_path = run_dir / "sample" / f"fake_images_{epoch}.png"
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"Missing formal evaluation for {name}: {evaluation_path}")
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing fixed-noise sample for {name}: {sample_path}")
        experiments[name] = {
            "run_dir": str(run_dir),
            "epoch": epoch,
            "evaluation_path": str(evaluation_path),
            "sample_path": str(sample_path),
            "metrics": read_json(evaluation_path),
        }
    return experiments


def draw_metric_chart(experiments, output_path):
    width, height = 1500, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    names = list(experiments)
    baseline_metrics = experiments["Base v3"]["metrics"]
    draw.text((40, 24), "Feature enhancement: same-protocol formal metrics", fill="black", font=font)
    draw.text(
        (40, 48),
        "KID/FID: lower is better | LPIPS/Laplacian: higher is better",
        fill="#444444",
        font=font,
    )

    panel_width = 700
    panel_height = 385
    for metric_index, (label, key, higher_is_better) in enumerate(METRICS):
        panel_x = 40 + (metric_index % 2) * 740
        panel_y = 90 + (metric_index // 2) * 420
        values = [experiments[name]["metrics"][key] for name in names]
        maximum = max(values) * 1.12 or 1.0
        draw.rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            outline="#cccccc",
            width=2,
        )
        draw.text((panel_x + 16, panel_y + 14), label, fill="black", font=font)
        baseline = baseline_metrics[key]
        for row, (name, value) in enumerate(zip(names, values)):
            y = panel_y + 58 + row * 62
            bar_x = panel_x + 150
            bar_width = int(410 * value / maximum)
            draw.text((panel_x + 16, y + 8), name, fill="black", font=font)
            draw.rectangle((bar_x, y, bar_x + bar_width, y + 28), fill=COLORS[name])
            change = percent_change(value, baseline)
            improved = change > 0 if higher_is_better else change < 0
            delta_color = "#047857" if improved else ("#555555" if abs(change) < 1e-9 else "#b91c1c")
            draw.text((bar_x + 420, y + 8), f"{value:.4f}", fill="black", font=font)
            draw.text(
                (bar_x + 505, y + 8),
                "Base" if name == "Base v3" else f"{change:+.1f}%",
                fill=delta_color,
                font=font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_sample_comparison(experiments, output_path):
    tile_size = 260
    label_height = 44
    margin = 20
    width = margin + len(experiments) * (tile_size + margin)
    height = margin + label_height + tile_size + margin
    output = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for column, (name, experiment) in enumerate(experiments.items()):
        x = margin + column * (tile_size + margin)
        label = f"{name} / epoch {experiment['epoch']}"
        draw.text((x, margin + 12), label, fill="black", font=font)
        with Image.open(experiment["sample_path"]) as sample:
            sample = sample.convert("RGB")
            sample.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), "#eeeeee")
            tile.paste(sample, ((tile_size - sample.width) // 2, (tile_size - sample.height) // 2))
            output.paste(tile, (x, margin + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def draw_se_multiseed_chart(summary, output_path):
    width, height = 1500, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 24), "SE attention multiseed validation", fill="black", font=font)
    draw.text(
        (40, 48),
        "Paired Base/SE results; SE checkpoints are selected per seed (200/100/100)",
        fill="#444444",
        font=font,
    )
    metric_specs = (
        ("KID (lower is better)", "kid_mean"),
        ("FID (lower is better)", "fid"),
        ("LPIPS diversity (higher is better)", "lpips_diversity"),
        ("Valid Laplacian (higher is better)", "laplacian_valid"),
    )
    for metric_index, (label, key) in enumerate(metric_specs):
        panel_x = 40 + (metric_index % 2) * 740
        panel_y = 90 + (metric_index // 2) * 420
        panel_width, panel_height = 700, 385
        rows = [
            (str(item["seed"]), item["base"][key], item["se"][key])
            for item in summary["per_seed"]
        ]
        rows.append(("mean", summary["base"][key]["mean"], summary["a1_se"][key]["mean"]))
        maximum = max(max(base, se) for _, base, se in rows) * 1.12 or 1.0
        draw.rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            outline="#cccccc",
            width=2,
        )
        draw.text((panel_x + 16, panel_y + 14), label, fill="black", font=font)
        for row, (seed, base, se) in enumerate(rows):
            y = panel_y + 66 + row * 70
            bar_x = panel_x + 90
            base_width = int(430 * base / maximum)
            se_width = int(430 * se / maximum)
            draw.text((panel_x + 16, y + 14), seed, fill="black", font=font)
            draw.rectangle((bar_x, y, bar_x + base_width, y + 18), fill=COLORS["Base v3"])
            draw.rectangle((bar_x, y + 24, bar_x + se_width, y + 42), fill=COLORS["SE attention"])
            draw.text((bar_x + 440, y + 2), f"Base {base:.4f}", fill="#555555", font=font)
            draw.text((bar_x + 440, y + 26), f"SE {se:.4f}", fill="#1d4ed8", font=font)
        change = summary["relative_change_of_means_percent"][key]
        draw.text(
            (panel_x + 525, panel_y + 14),
            f"SE mean vs Base: {change:+.2f}%",
            fill="#b91c1c" if key in {"kid_mean", "fid"} and change > 0 else "#047857",
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_summary(experiments):
    baseline = experiments["Base v3"]["metrics"]
    summary = {"baseline": "Base v3", "experiments": {}}
    for name, experiment in experiments.items():
        result = {
            "run_dir": experiment["run_dir"],
            "epoch": experiment["epoch"],
            "metrics": {},
        }
        for label, key, higher_is_better in METRICS:
            value = experiment["metrics"][key]
            change = percent_change(value, baseline[key])
            result["metrics"][key] = {
                "label": label,
                "value": value,
                "change_vs_base_percent": change,
                "improved": change > 0 if higher_is_better else change < 0,
            }
        summary["experiments"][name] = result
    return summary


def main(args=None):
    parsed = parse_args(args)
    experiments = load_experiments(parsed.attention_run, parsed.attention_epoch)
    output_dir = resolve(parsed.output_dir)
    draw_metric_chart(experiments, output_dir / "metrics_comparison.png")
    draw_sample_comparison(experiments, output_dir / "sample_comparison.png")
    if parsed.se_multiseed_summary:
        draw_se_multiseed_chart(
            read_json(resolve(parsed.se_multiseed_summary)),
            output_dir / "se_multiseed_comparison.png",
        )
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(build_summary(experiments), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Feature comparison saved to {output_dir}")


if __name__ == "__main__":
    main()
