"""
数据集准备工具
将SSDD和SeaShips数据集转换为YOLO格式，统一使用旋转框格式（6参数）
"""
import os
import shutil
from pathlib import Path
from typing import Tuple, List
from Config.config import Config


class DatasetPreparer:
    """数据集准备器，统一输出旋转框格式（6参数）"""

    def __init__(self, output_root: str = None):
        """
        初始化准备器

        Args:
            output_root: 输出根目录，默认使用配置中的路径
        """
        self.output_root = Path(output_root or Config.OUTPUT_ROOT)
        self.train_img_dir = self.output_root / "images/train"
        self.val_img_dir = self.output_root / "images/val"
        self.train_label_dir = self.output_root / "labels/train"
        self.val_label_dir = self.output_root / "labels/val"

        self._create_directories()

    def _create_directories(self) -> None:
        """创建输出目录"""
        for d in [self.train_img_dir, self.val_img_dir,
                  self.train_label_dir, self.val_label_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_and_convert_label(line: str) -> str:
        """
        解析并转换单个标签行为统一旋转框格式（6参数）

        统一输出格式: class_id x_center y_center width height angle
        - 如果是旋转框，保留原始角度
        - 如果是水平框，角度设为0

        Args:
            line: 标签行文本

        Returns:
            转换后的标签行（6参数格式），如果无效则返回空字符串
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return ""

        # 判断是否为旋转框 (6个参数)
        is_rotated = len(parts) >= 6

        if is_rotated:
            # 已经是旋转框，保留格式（类别统一为0）
            new_parts = ["0"] + parts[1:6]
        else:
            # 水平框转旋转框（角度设为0）
            new_parts = ["0"] + parts[1:5] + ["0"]

        return " ".join(new_parts)

    def _copy_and_convert(
        self,
        img_dir: str,
        label_dir: str,
        out_img_dir: Path,
        out_label_dir: Path,
        prefix: str
    ) -> int:
        """
        拷贝图像并转换标签为单类别旋转框格式

        Args:
            img_dir: 源图像目录
            label_dir: 源标签目录
            out_img_dir: 输出图像目录
            out_label_dir: 输出标签目录
            prefix: 文件名前缀

        Returns:
            处理的样本数
        """
        img_path = Path(img_dir)
        label_path = Path(label_dir)

        if not img_path.exists():
            print(f"⚠️ 图像目录不存在: {img_dir}")
            return 0

        if not label_path.exists():
            print(f"⚠️ 标签目录不存在: {label_dir}")
            return 0

        count = 0
        rotated_count = 0
        horizontal_count = 0

        for img_file in img_path.glob("*.jpg"):
            name = img_file.stem
            src_label = label_path / f"{name}.txt"

            if not src_label.exists():
                continue

            new_name = f"{prefix}_{name}"

            # 拷贝图像
            shutil.copy(img_file, out_img_dir / f"{new_name}.jpg")

            # 转换标签（统一为ship类别0，旋转框格式）
            new_labels = []
            with open(src_label, "r") as f:
                for line in f:
                    new_label = self._parse_and_convert_label(line)
                    if new_label:
                        new_labels.append(new_label)
                        # 统计旋转框/水平框
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            rotated_count += 1
                        else:
                            horizontal_count += 1

            if not new_labels:
                continue

            with open(out_label_dir / f"{new_name}.txt", "w") as f:
                f.write("\n".join(new_labels))

            count += 1

        if rotated_count > 0 or horizontal_count > 0:
            print(f"   {prefix}: 处理 {count} 张图像, {rotated_count} 个旋转框, {horizontal_count} 个水平框(角度设为0)")

        return count

    def prepare(self) -> Tuple[int, int]:
        """
        准备完整数据集

        Returns:
            (训练集样本数, 验证集样本数)
        """
        print("========== 构建 YOLO 船舶数据集 ==========")
        print("输出格式: 统一旋转框 (6参数: x_center y_center width height angle)")
        print("-" * 40)

        train_count = 0
        val_count = 0

        # SSDD训练集 - 近岸
        train_count += self._copy_and_convert(
            Config.SSDD_TRAIN_INSHORE_IMG,
            Config.SSDD_TRAIN_LABEL,
            self.train_img_dir,
            self.train_label_dir,
            "ssdd_inshore"
        )

        # SSDD训练集 - 远岸
        train_count += self._copy_and_convert(
            Config.SSDD_TRAIN_OFFSHORE_IMG,
            Config.SSDD_TRAIN_LABEL,
            self.train_img_dir,
            self.train_label_dir,
            "ssdd_offshore"
        )

        # SSDD验证集 - 近岸测试
        val_count += self._copy_and_convert(
            Config.SSDD_TEST_INSHORE_IMG,
            Config.SSDD_TEST_INSHORE_LABEL,
            self.val_img_dir,
            self.val_label_dir,
            "ssdd_test_inshore"
        )

        # SSDD验证集 - 远岸测试
        val_count += self._copy_and_convert(
            Config.SSDD_TEST_OFFSHORE_IMG,
            Config.SSDD_TEST_OFFSHORE_LABEL,
            self.val_img_dir,
            self.val_label_dir,
            "ssdd_test_offshore"
        )

        # SeaShips作为训练集补充
        train_count += self._copy_and_convert(
            Config.SEASHIP_IMG,
            Config.SEASHIP_LABEL,
            self.train_img_dir,
            self.train_label_dir,
            "seaship"
        )

        print("-" * 40)
        print(f"训练集样本数: {train_count}")
        print(f"验证集样本数: {val_count}")
        print("输出格式: 统一旋转框 (6参数: x_center y_center width height angle)")
        print("✅ YOLO 数据集构建完成")
        print("-" * 40)
        print(f"数据集路径: {self.output_root}")

        return train_count, val_count


def prepare_ship_dataset(output_root: str = None) -> Tuple[int, int]:
    """
    便捷的数据集准备函数

    Args:
        output_root: 输出根目录

    Returns:
        (训练集样本数, 验证集样本数)
    """
    preparer = DatasetPreparer(output_root)
    return preparer.prepare()


if __name__ == "__main__":
    # 验证路径配置
    valid, errors = Config.validate_paths()
    if not valid:
        print("⚠️ 路径验证失败:")
        for error in errors:
            print(f"   - {error}")
        print("\n提示: 可以通过环境变量覆盖默认路径，例如:")
        print("   export SSDD_TRAIN_INSHORE_IMG=/your/path/to/images")
    else:
        prepare_ship_dataset()
