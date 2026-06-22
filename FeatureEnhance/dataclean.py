import cv2
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =========================
# 配置
# =========================

INPUT_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_cleaned")

# 删除最模糊的比例
REMOVE_RATIO = 0.10

# 支持格式
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# 重复运行时先清理上一次生成的图片，避免旧文件残留到清洗后的数据集。
CLEAR_OUTPUT_DIR = True

# =========================
# 计算清晰度
# =========================

def laplacian_variance(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(score)


def list_images(input_dir):
    image_files = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(image_files)


def prepare_output_dir(output_dir):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CLEAR_OUTPUT_DIR:
        return

    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            path.unlink()


# =========================
# 主程序
# =========================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {INPUT_DIR}")

    image_files = list_images(INPUT_DIR)

    print(f"发现 {len(image_files)} 张图片")
    if not image_files:
        raise ValueError(f"未找到支持格式的图片: {INPUT_DIR}")

    results = []

    print("计算清晰度评分...")

    for img_path in tqdm(image_files):

        score = laplacian_variance(img_path)

        if score is None:
            continue

        results.append({
            "filename": img_path.name,
            "path": str(img_path),
            "score": score
        })

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("所有图片都无法读取，未生成清洗结果。")

    df = df.sort_values(
        by="score",
        ascending=True
    )

    # =========================
    # 删除最模糊10%
    # =========================

    remove_count = int(
        len(df) * REMOVE_RATIO
    )

    threshold_index = max(min(remove_count - 1, len(df) - 1), 0)
    threshold = df.iloc[threshold_index]["score"]

    print(
        f"\n删除比例: {REMOVE_RATIO*100:.1f}%"
    )

    print(
        f"阈值: {threshold:.2f}"
    )

    removed_df = df.iloc[:remove_count]

    kept_df = df.iloc[remove_count:]

    print(
        f"删除图片数: {len(removed_df)}"
    )

    print(
        f"保留图片数: {len(kept_df)}"
    )

    # Do not clear a previous valid output until the new input has been read and scored.
    prepare_output_dir(OUTPUT_DIR)

    # =========================
    # 保存保留图片
    # =========================

    print("\n复制清洗后数据集...")

    for _, row in tqdm(
        kept_df.iterrows(),
        total=len(kept_df)
    ):

        src = Path(row["path"])

        dst = OUTPUT_DIR / src.name

        shutil.copy2(src, dst)

    # =========================
    # 保存报告
    # =========================

    df.to_csv(
        "blur_scores.csv",
        index=False
    )

    removed_df.to_csv(
        "removed_images.csv",
        index=False
    )

    kept_df.to_csv(
        "kept_images.csv",
        index=False
    )

    print("\n完成")
    print(f"清洗后数据集保存至: {OUTPUT_DIR}")
    print("blur_scores.csv 已保存")


if __name__ == "__main__":
    main()
