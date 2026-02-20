"""
数据集准备模块 - 简化版
自动处理数据集并统一为旋转框格式
"""
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random

from Config.config import Config


class DatasetPreparer:
    """数据集准备器"""

    def __init__(self, output_root: Path = None):
        self.output_root = output_root or Config.get_output_dir("dataset")
        self.train_dir = self.output_root / "train"
        self.val_dir = self.output_root / "val"
        self.test_dir = self.output_root / "test"

        for d in [self.train_dir, self.val_dir, self.test_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """准备完整数据集"""
        print("\n" + "=" * 60)
        print("📦 准备数据集")
        print("=" * 60)

        # 收集所有数据
        all_data = self._collect_all_data()
        print(f"   找到 {len(all_data)} 个样本")

        # 划分数据集
        random.shuffle(all_data)
        n = len(all_data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]

        print(f"   训练集: {len(train_data)}")
        print(f"   验证集: {len(val_data)}")
        print(f"   测试集: {len(test_data)}")

        # 复制文件
        self._copy_files(train_data, self.train_dir)
        self._copy_files(val_data, self.val_dir)
        self._copy_files(test_data, self.test_dir)

        print(f"\n✅ 数据集准备完成: {self.output_root}")
        return str(self.output_root)

    def _collect_all_data(self) -> List[Tuple[Path, Path]]:
        """收集所有数据"""
        data_pairs = []

        # SSDD 数据
        if Config.SSDD_TRAIN_INSHORE_IMG.exists():
            data_pairs.extend(self._collect_from_dir(
                Config.SSDD_TRAIN_INSHORE_IMG,
                Config.SSDD_TRAIN_LABEL
            ))

        if Config.SSDD_TRAIN_OFFSHORE_IMG.exists():
            data_pairs.extend(self._collect_from_dir(
                Config.SSDD_TRAIN_OFFSHORE_IMG,
                Config.SSDD_TRAIN_LABEL
            ))

        # SeaShips 数据
        if Config.SEASHIP_IMG.exists():
            data_pairs.extend(self._collect_from_dir(
                Config.SEASHIP_IMG,
                Config.SEASHIP_LABEL
            ))

        return data_pairs

    def _collect_from_dir(self, img_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
        """从目录收集数据对"""
        pairs = []

        for img_path in img_dir.glob("*.jpg"):
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))

        return pairs

    def _copy_files(self, data_pairs: List[Tuple[Path, Path]], target_dir: Path):
        """复制文件到目标目录"""
        for img_path, label_path in data_pairs:
            # 复制图片
            shutil.copy2(img_path, target_dir / "images" / img_path.name)

            # 处理并复制标签
            self._process_label(label_path, target_dir / "labels" / label_path.name)

    def _process_label(self, src_label: Path, dst_label: Path):
        """处理标签文件 - 统一为旋转框格式"""
        with open(src_label, 'r') as f:
            lines = f.readlines()

        processed_lines = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) == 5:
                # 水平框: class x_center y_center width height
                cls_id = parts[0]
                x, y, w, h = map(float, parts[1:5])
                # 转换为旋转框格式，角度设为0
                processed_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} 0.0\n")
            elif len(parts) == 6:
                # 旋转框: class x_center y_center width height angle
                processed_lines.append(line)
            else:
                # 其他格式，尝试解析
                cls_id = parts[0]
                coords = list(map(float, parts[1:]))
                if len(coords) >= 4:
                    x, y, w, h = coords[:4]
                    angle = coords[4] if len(coords) > 4 else 0.0
                    processed_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {angle:.6f}\n")

        with open(dst_label, 'w') as f:
            f.writelines(processed_lines)


def prepare_dataset():
    """准备数据集的便捷函数"""
    preparer = DatasetPreparer()
    return preparer.prepare_dataset()


if __name__ == "__main__":
    prepare_dataset()
