"""
数据集准备模块 
统一数据集结构：
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

标签格式：Ultralytics YOLO OBB格式 (class x1 y1 x2 y2 x3 y3 x4 y4)
- 4个角点坐标（归一化0-1）
- 角点顺序：左上、右上、右下、左下（顺时针）
"""
import math
from pathlib import Path
from typing import Dict, List, Tuple

from Config.config import Config


def xywha_to_xyxyxyxy(class_id: int, x_center: float, y_center: float,
                       width: float, height: float, angle_deg: float) -> Tuple:
    """
    将 (x_center, y_center, width, height, angle) 转换为四个角点 (x1,y1,x2,y2,x3,y3,x4,y4)

    Args:
        class_id: 类别ID
        x_center: 中心点 x (归一化 0-1)
        y_center: 中心点 y (归一化 0-1)
        width: 宽度 (归一化 0-1)
        height: 高度 (归一化 0-1)
        angle_deg: 旋转角度（度）

    Returns:
        (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
    """
    # 角度转弧度
    angle_rad = math.radians(angle_deg)

    # 半宽半高
    w2 = width / 2
    h2 = height / 2

    # 计算四个角点（未旋转前，相对于中心）
    # 顺序：左上、右上、右下、左下（顺时针）
    corners = [
        (-w2, -h2),  # 左上
        (w2, -h2),   # 右上
        (w2, h2),    # 右下
        (-w2, h2),   # 左下
    ]

    # 旋转并平移
    rotated_corners = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    for dx, dy in corners:
        # 旋转
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        # 平移到中心
        x = x_center + rx
        y = y_center + ry
        rotated_corners.extend([x, y])

    return (class_id, *rotated_corners)


def convert_label_line(line: str) -> str:
    """
    转换单行标签格式

    Args:
        line: 输入标签行

    Returns:
        转换后的标签行（Ultralytics OBB格式）
    """
    parts = line.strip().split()

    if len(parts) == 6:
        # 6列格式：class x_center y_center width height angle
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        angle = float(parts[5])

        # 转换为 OBB 格式
        converted = xywha_to_xyxyxyxy(class_id, x_center, y_center, width, height, angle)
        return f"{converted[0]} {converted[1]:.6f} {converted[2]:.6f} {converted[3]:.6f} {converted[4]:.6f} {converted[5]:.6f} {converted[6]:.6f} {converted[7]:.6f} {converted[8]:.6f}"

    elif len(parts) == 5:
        # 5列格式：class x_center y_center width height（水平框，角度为0）
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # 转换为 OBB 格式（角度为0）
        converted = xywha_to_xyxyxyxy(class_id, x_center, y_center, width, height, 0.0)
        return f"{converted[0]} {converted[1]:.6f} {converted[2]:.6f} {converted[3]:.6f} {converted[4]:.6f} {converted[5]:.6f} {converted[6]:.6f} {converted[7]:.6f} {converted[8]:.6f}"

    elif len(parts) == 9:
        # 已经是 OBB 格式，直接返回
        return line.strip()

    else:
        # 格式不正确，返回空字符串
        return ""


def convert_label_file(input_path: Path, output_path: Path = None) -> bool:
    """
    转换单个标签文件

    Args:
        input_path: 输入标签文件路径
        output_path: 输出标签文件路径（默认为输入路径，即覆盖原文件）

    Returns:
        是否成功转换
    """
    if output_path is None:
        output_path = input_path

    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        converted_lines = []
        for line in lines:
            converted = convert_label_line(line)
            if converted:
                converted_lines.append(converted)

        # 写入输出文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            if converted_lines:
                f.write('\n'.join(converted_lines) + '\n')

        return True

    except Exception as e:
        print(f"    错误: 转换 {input_path.name} 失败: {e}")
        return False


class DatasetPreparer:
    """数据集验证器 - 验证人工准备好的数据集"""

    def __init__(self, dataset_root: Path = None):
        """
        初始化数据集验证器

        Args:
            dataset_root: 数据集根目录，默认为项目目录下的dataset文件夹
        """
        self.dataset_root = dataset_root or Config.DATASET_ROOT
        # 新结构：images/ 和 labels/ 目录下有 train/val/test 子目录
        self.train_img_dir = self.dataset_root / "images" / "train"
        self.train_label_dir = self.dataset_root / "labels" / "train"
        self.val_img_dir = self.dataset_root / "images" / "val"
        self.val_label_dir = self.dataset_root / "labels" / "val"
        self.test_img_dir = self.dataset_root / "images" / "test"
        self.test_label_dir = self.dataset_root / "labels" / "test"

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
            "train": self._validate_split("train", self.train_img_dir, self.train_label_dir),
            "val": self._validate_split("val", self.val_img_dir, self.val_label_dir),
            "test": self._validate_split("test", self.test_img_dir, self.test_label_dir),
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
            print("   ├── images/")
            print("   │   ├── train/")
            print("   │   ├── val/")
            print("   │   └── test/")
            print("   └── labels/")
            print("       ├── train/")
            print("       ├── val/")
            print("       └── test/")
        else:
            print(f"\n✅ 数据集验证通过")
            print(f"   总样本数: {total_samples}")
            print(f"   训练集: {result['train']['count']}")
            print(f"   验证集: {result['val']['count']}")
            print(f"   测试集: {result['test']['count']}")

        return result

    def _validate_split(self, split_name: str, img_dir: Path, label_dir: Path) -> Dict[str, any]:
        """
        验证数据集划分（train/val/test）

        Args:
            split_name: 划分名称
            img_dir: 图片目录
            label_dir: 标签目录

        Returns:
            验证结果
        """
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

    def convert_labels(self) -> Dict[str, int]:
        """
        转换所有标签为 Ultralytics OBB 格式

        Returns:
            转换统计信息
        """
        print("\n" + "=" * 60)
        print("🔄 转换标签格式为 Ultralytics OBB 格式")
        print("=" * 60)

        stats = {"train": 0, "val": 0, "test": 0}

        splits = [
            ("train", self.train_label_dir),
            ("val", self.val_label_dir),
            ("test", self.test_label_dir),
        ]

        for split_name, label_dir in splits:
            if not label_dir.exists():
                print(f"   ⚠️  {split_name}: 标签目录不存在")
                continue

            label_files = list(label_dir.glob("*.txt"))
            if not label_files:
                print(f"   ⚠️  {split_name}: 未找到标签文件")
                continue

            print(f"   📝 {split_name}: 找到 {len(label_files)} 个标签文件")

            converted_count = 0
            for label_file in label_files:
                # 跳过备份文件
                if label_file.suffix == '.backup':
                    continue

                # 检查是否需要转换
                needs_conversion = False
                try:
                    with open(label_file, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            parts = first_line.split()
                            # 如果是5列或6列格式，需要转换
                            if len(parts) in [5, 6]:
                                needs_conversion = True
                            # 如果已经是9列，不需要转换
                            elif len(parts) == 9:
                                needs_conversion = False
                except:
                    continue

                if needs_conversion:
                    # 备份原文件
                    backup_path = label_file.with_suffix('.txt.backup')
                    if not backup_path.exists():
                        label_file.rename(backup_path)

                    # 转换
                    if convert_label_file(backup_path, label_file):
                        converted_count += 1
                else:
                    converted_count += 1  #  already in correct format

            print(f"   ✅ {split_name}: 成功处理 {converted_count} 个文件")
            stats[split_name] = converted_count

        total = sum(stats.values())
        print(f"\n✅ 总共处理 {total} 个标签文件")
        if total > 0:
            print("   原文件已备份为 .txt.backup")
        print("=" * 60)

        return stats

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
        for label_dir in [self.train_label_dir, self.val_label_dir, self.test_label_dir]:
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
                    if len(parts) == 9:
                        # OBB格式: class x1 y1 x2 y2 x3 y3 x4 y4
                        # 计算角度（简化处理，通过比较对角线角度）
                        x1, y1 = float(parts[1]), float(parts[2])
                        x2, y2 = float(parts[3]), float(parts[4])
                        x3, y3 = float(parts[5]), float(parts[6])

                        # 计算中心点
                        cx = (x1 + x2 + x3 + float(parts[7])) / 4
                        cy = (y1 + y2 + y3 + float(parts[8])) / 4

                        # 计算角度（使用长边）
                        dx1 = x2 - x1
                        dy1 = y2 - y1
                        dx2 = x3 - x2
                        dy2 = y3 - y2

                        # 选择较长的边计算角度
                        len1 = math.sqrt(dx1**2 + dy1**2)
                        len2 = math.sqrt(dx2**2 + dy2**2)

                        if len1 > len2:
                            angle = math.degrees(math.atan2(dy1, dx1))
                        else:
                            angle = math.degrees(math.atan2(dy2, dx2))

                        angles.append(angle)
                        total_objects += 1

                        if abs(angle) < 5:  # 接近0视为水平框
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

    # 验证数据集结构
    result = preparer.validate_dataset()

    if result["valid"]:
        # 转换标签格式
        preparer.convert_labels()

        # 分析标签
        preparer.analyze_labels()

    return result


if __name__ == "__main__":
    validate_dataset()
