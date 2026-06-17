import os
import torch
from torchvision.utils import save_image
from model64 import Generator, Discriminator

# =====================
# 参数
# =====================

candidate_num = 10000   # 先生成10000张
save_num = 100          # 保存前100张

# =====================
# Device
# =====================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# =====================
# 输出目录
# =====================

save_dir = "genImage"
os.makedirs(save_dir, exist_ok=True)

# 当前最大编号
existing_ids = []

for f in os.listdir(save_dir):
    if f.startswith("generated_") and f.endswith(".png"):
        try:
            num = int(f.split("_")[1].split(".")[0])
            existing_ids.append(num)
        except:
            pass

start_id = max(existing_ids, default=0) + 1

# =====================
# 加载模型
# =====================

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.load_state_dict(
    torch.load("generator.pth", map_location=device)
)

discriminator.load_state_dict(
    torch.load("discriminator.pth", map_location=device)
)

generator.eval()
discriminator.eval()

# =====================
# 生成候选图片
# =====================

candidates = []

with torch.no_grad():

    for i in range(candidate_num):

        noise = torch.randn(
            1,
            100,
            1,
            1,
            device=device
        )

        img = generator(noise)

        score = discriminator(img).item()

        candidates.append(
            (score, img.cpu())
        )

        if (i + 1) % 1000 == 0:
            print(
                f"Generated {i+1}/{candidate_num}"
            )

# =====================
# 排序
# =====================

print("Sorting...")

candidates.sort(
    key=lambda x: x[0],
    reverse=True
)

# =====================
# 保存前100张
# =====================

for i in range(save_num):

    score, img = candidates[i]

    img = (img + 1) / 2

    save_path = os.path.join(
        save_dir,
        f"generated_{start_id+i:04d}.png"
    )

    save_image(img, save_path)

    print(
        f"Saved {save_path} | Score={score:.4f}"
    )

print("\nFinished!")
print(f"Top {save_num} images saved.")