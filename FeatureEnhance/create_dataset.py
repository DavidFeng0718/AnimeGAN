import torch
from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid
import torch.utils.data.dataset as Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class My_dataset(Dataset.Dataset):

    def __init__(self, path, transform):

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.path}")

        self.image_paths = sorted(
            item for item in self.path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise ValueError(f"No supported images found in dataset path: {self.path}")

        self.transform = transform

    def __getitem__(self, index):

        loc_data = self.image_paths[index]
        with Image.open(loc_data) as image:
            data = image.convert("RGB")
        data = self.transform(data)
        return data

    def __len__(self):

        return len(self.image_paths)


def save_img(tensor, fp):

    # The tensors are normalized to [-1, 1]. A white separator makes it clear
    # that grid lines are preview UI rather than generated image content.
    grid = make_grid(tensor, padding=2, pad_value=1.0)
    ndarr = grid.clamp(-1.0, 1.0).add(1.0).mul(127.5).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)

