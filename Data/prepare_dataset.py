"""
数据集准备模块 - 简化版
假设数据集已经人工准备好，此模块仅提供验证和统计功能

统一数据集结构（人工准备）：
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

标签格式：YOLO OBB格式 (class x_center y_center width height angle)
- 角度单位：度（degrees）
- 水平框角度为0
"""
from pathlib import Path
from typing import Dict, List, Tuple
import math

from Config.config import Config


class DatasetPreparer:
    """数据集验证器 - 验证人工准备好的数据集"""

    def __init__(self, dataset_root: Path = None):
        """
        初始化数据集验证器

        Args:
            dataset_root: 数据集根目录，默认为项目目录下的dataset文件夹
        """
        self.dataset_root = dataset_root or Config.DATASET_ROOT
        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "val"
        self.test_dir = self.dataset_root / "test"

    def validate_dataset(self) -> Dict[str, any]:
        """
        验证数据集结构完整性

        Returns:
            验证结果字典
        """
        print("\n" + "=" * 60)
        print("📁 验证数据集")
        print("=" * 60)
        print(f"   数据集路径: {self.dataset_root}")

        result = {
            "valid": True,
            "train": self._validate_split("train", self.train_dir),
            "val": self._validate_split("val", self.val_dir),
            "test": self._validate_split("test", self.test_dir),
        }

        # 检查是否有有效数据
        total_samples = (result["train"]["count"] +
                        result["val"]["count"] +
                        result["test"]["count"])

        if total_samples == 0:
            result["valid"] = False
            print("\n   ❌ 错误: 未找到任何数据样本")
            print("   请确保数据集已准备好，结构如下:")
            print("   dataset/")
            print("   ├── train/images/ 和 train/labels/")
            print("   ├── val/images/ 和 val/labels/")
            print("   └── test/images/ 和 test/labels/")
        else:
            print(f"\n✅ 数据集验证通过")
            print(f"   总样本数: {total_samples}")
            print(f"   训练集: {result['train']['count']}")
            print(f"   验证集: {result['val']['count']}")
            print(f"   测试集: {result['test']['count']}")

        return result

    def _validate_split(self, split_name: str, split_dir: Path) -> Dict[str, any]:
        """
        验证数据集划分（train/val/test）

        Args:
            split_name: 划分名称
            split_dir: 划分目录

        Returns:
            验证结果
        """
        img_dir = split_dir / "images"
        label_dir = split_dir / "labels"

        result = {
            "exists": False,
            "count": 0,
            "images_dir": str(img_dir),
            "labels_dir": str(label_dir),
        }

        if not img_dir.exists():
            print(f"   ⚠️  {split_name}: 图片目录不存在 - {img_dir}")
            return result

        if not label_dir.exists():
            print(f"   ⚠️  {split_name}: 标签目录不存在 - {label_dir}")
            return result

        # 统计图片和标签
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        label_files = list(label_dir.glob("*.txt"))

        # 匹配的图片-标签对
        matched = 0
        for img_path in img_files:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                matched += 1

        result["exists"] = True
        result["count"] = matched

        if matched > 0:
            print(f"   ✓ {split_name}: {matched} 个样本")
        else:
            print(f"   ⚠️  {split_name}: 目录存在但无匹配样本")

        return result

    def analyze_labels(self, max_samples: int = 100) -> Dict[str, any]:
        """
        分析标签格式和统计信息

        Args:
            max_samples: 最大分析样本数

        Returns:
            分析结果
        """
        print("\n" + "=" * 60)
        print("📊 分析标签格式")
        print("=" * 60)

        stats = {
            "total_labels": 0,
            "horizontal_boxes": 0,  # 角度为0的框
            "rotated_boxes": 0,     # 角度不为0的框
            "avg_objects_per_image": 0,
            "angle_range": {"min": float('inf'), "max": float('-inf')},
        }

        label_files = []
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            label_dir = split_dir / "labels"
            if label_dir.exists():
                label_files.extend(list(label_dir.glob("*.txt")))

        if not label_files:
            print("   ⚠️  未找到标签文件")
            return stats

        # 限制分析样本数
        label_files = label_files[:max_samples]

        total_objects = 0
        angles = []

        for label_path in label_files:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        # YOLO OBB格式: class x y w h angle
                        angle = float(parts[5])
                        angles.append(angle)
                        total_objects += 1

                        if abs(angle) < 0.1:  # 接近0视为水平框
                            stats["horizontal_boxes"] += 1
                        else:
                            stats["rotated_boxes"] += 1

            except Exception as e:
                continue

        stats["total_labels"] = len(label_files)
        stats["avg_objects_per_image"] = total_objects / len(label_files) if label_files else 0

        if angles:
            stats["angle_range"]["min"] = min(angles)
            stats["angle_range"]["max"] = max(angles)

        print(f"   分析样本: {len(label_files)}")
        print(f"   总目标数: {total_objects}")
        print(f"   平均每图目标数: {stats['avg_objects_per_image']:.2f}")
        print(f"   水平框: {stats['horizontal_boxes']}")
        print(f"   旋转框: {stats['rotated_boxes']}")
        if angles:
            print(f"   角度范围: {stats['angle_range']['min']:.2f}° ~ {stats['angle_range']['max']:.2f}°")

        return stats


def validate_dataset():
    """验证数据集的便捷函数"""
    preparer = DatasetPreparer()
    result = preparer.validate_dataset()

    if result["valid"]:
        preparer.analyze_labels()

    return result


if __name__ == "__main__":
    validate_dataset()
