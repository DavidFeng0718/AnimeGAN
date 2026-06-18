import torch
from torchvision.utils import save_image

from model64 import Generator

device = torch.device("mps")

# 加载checkpoint
ckpt = torch.load(
    "checkpoint_epoch_100.pt",
    map_location=device
)

# 创建Generator
generator = Generator(noise_dim=100).to(device)

# 加载权重
generator.load_state_dict(
    ckpt["generator_state_dict"]
)

generator.eval()

import os

# 创建输出目录
os.makedirs("test", exist_ok=True)

with torch.no_grad():
    for i in range(100):
        z = torch.randn(
            1,
            100,
            1,
            1,
            device=device
        )

        fake = generator(z)
        fake = (fake + 1) / 2

        save_image(fake, f"test/image_{i:03d}.png")

print("100 images saved to test/")