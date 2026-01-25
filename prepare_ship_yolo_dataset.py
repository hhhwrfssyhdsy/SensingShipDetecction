import os
import shutil
from pathlib import Path
from Config.DataReading import *

# =========================
# 输出 YOLO 数据集
# =========================

OUTPUT_ROOT = r"/root/ShipDetection_improved"

TRAIN_IMG_DIR = os.path.join(OUTPUT_ROOT, "images/train")
VAL_IMG_DIR = os.path.join(OUTPUT_ROOT, "images/val")
TRAIN_LAB_DIR = os.path.join(OUTPUT_ROOT, "labels/train")
VAL_LAB_DIR = os.path.join(OUTPUT_ROOT, "labels/val")

for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LAB_DIR, VAL_LAB_DIR]:
    os.makedirs(d, exist_ok=True)

def copy_pairs(img_dir, label_dir, out_img_dir, out_lab_dir, prefix):
    """
    拷贝图片，并将所有船舶统一为单类别 ship (class 0)
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    count = 0

    for img_path in img_dir.glob("*.jpg"):
        name = img_path.stem
        label_path = label_dir / f"{name}.txt"

        if not label_path.exists():
            continue

        new_name = f"{prefix}_{name}"

        # ---------- copy image ----------
        shutil.copy(
            img_path,
            os.path.join(out_img_dir, new_name + ".jpg")
        )

        # ---------- process label ----------
        new_label_lines = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                # 强制统一为 ship 类
                parts = ["0"] + parts[1:5]

                new_label_lines.append(" ".join(parts))

        if len(new_label_lines) == 0:
            continue

        with open(os.path.join(out_lab_dir, new_name + ".txt"), "w") as f:
            f.write("\n".join(new_label_lines))

        count += 1

    return count


print("========== 构建 YOLO 船舶数据集 ==========")

train_count = 0
val_count = 0

# SSDD train
train_count += copy_pairs(
    SSDD_train_inshore_img_path,
    SSDD_train_label_path,
    TRAIN_IMG_DIR,
    TRAIN_LAB_DIR,
    "ssdd_inshore"
)

train_count += copy_pairs(
    SSDD_train_offshore_img_path,
    SSDD_train_label_path,
    TRAIN_IMG_DIR,
    TRAIN_LAB_DIR,
    "ssdd_offshore"
)

# SSDD val
val_count += copy_pairs(
    SSDD_test_inshore_img_path,
    SSDD_test_inshore_label_path,
    VAL_IMG_DIR,
    VAL_LAB_DIR,
    "ssdd_test_inshore"
)

val_count += copy_pairs(
    SSDD_test_offshore_img_path,
    SSDD_test_offshore_label_path,
    VAL_IMG_DIR,
    VAL_LAB_DIR,
    "ssdd_test_offshore"
)

# SeaShips → 作为训练集
train_count += copy_pairs(
    S_img_path,
    S_label_path,
    TRAIN_IMG_DIR,
    TRAIN_LAB_DIR,
    "seaship"
)

print("----------------------------------------")
print(f"训练集样本数: {train_count}")
print(f"验证集样本数: {val_count}")
print("✅ YOLO 数据集构建完成")
print("----------------------------------------")
print(f"数据集路径: {OUTPUT_ROOT}")

