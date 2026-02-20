"""
数据集分析工具
用于分析船舶数据集中的目标分布，指导训练策略选择
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class ObjectStats:
    """目标统计信息"""
    total: int = 0
    small: int = 0
    medium: int = 0
    large: int = 0
    rotated: int = 0  # 旋转框数量
    horizontal: int = 0  # 水平框数量
    avg_size: float = 0.0
    marine_difficulty: float = 0.0

    @property
    def small_ratio(self) -> float:
        """小目标比例"""
        return self.small / self.total if self.total > 0 else 0.0

    @property
    def medium_ratio(self) -> float:
        """中目标比例"""
        return self.medium / self.total if self.total > 0 else 0.0

    @property
    def large_ratio(self) -> float:
        """大目标比例"""
        return self.large / self.total if self.total > 0 else 0.0

    @property
    def rotated_ratio(self) -> float:
        """旋转框比例"""
        return self.rotated / self.total if self.total > 0 else 0.0


class DatasetAnalyzer:
    """数据集分析器"""

    # 目标尺寸分类阈值（相对于图像面积的比例）
    SMALL_THRESHOLD = 0.005
    MEDIUM_THRESHOLD = 0.02

    def __init__(self, img_paths: List[str], label_paths: List[str]):
        """
        初始化分析器

        Args:
            img_paths: 图像目录路径列表
            label_paths: 标签目录路径列表（与img_paths对应）
        """
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.samples: List[Tuple[str, str]] = []
        self._collect_samples()

    def _collect_samples(self) -> None:
        """收集所有样本路径"""
        for img_dir, label_dir in zip(self.img_paths, self.label_paths):
            img_path = Path(img_dir)
            label_path = Path(label_dir)

            if not img_path.exists() or not label_path.exists():
                continue

            for img_file in img_path.glob("*.jpg"):
                label_file = label_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    self.samples.append((str(img_file), str(label_file)))

        print(f"📊 共收集 {len(self.samples)} 个样本")

    @staticmethod
    def parse_yolo_label(label_path: str, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析YOLO格式标签，支持水平框和旋转框

        YOLO格式:
        - 水平框: class_id x_center y_center width height (5个参数)
        - 旋转框: class_id x_center y_center width height angle (6个参数)

        Args:
            label_path: 标签文件路径
            image_size: 图像尺寸 (width, height)

        Returns:
            (boxes, is_rotated) - 边界框数组和是否为旋转框的标记
            boxes: 每行 [x_min, y_min, x_max, y_max] 或 [cx, cy, w, h, angle]
            is_rotated: 每个框是否为旋转框
        """
        boxes = []
        is_rotated_list = []

        if not os.path.exists(label_path):
            return np.array(boxes), np.array(is_rotated_list)

        img_w, img_h = image_size

        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) < 5:
                    continue

                # 判断是否为旋转框 (6个参数)
                has_angle = len(data) >= 6

                x_center = float(data[1])
                y_center = float(data[2])
                width = float(data[3])
                height = float(data[4])

                if has_angle:
                    # 旋转框: 保存原始参数 + 角度
                    angle = float(data[5])
                    # 转换为绝对坐标
                    cx_abs = x_center * img_w
                    cy_abs = y_center * img_h
                    w_abs = width * img_w
                    h_abs = height * img_h
                    # 存储: [cx, cy, w, h, angle]
                    boxes.append([cx_abs, cy_abs, w_abs, h_abs, angle])
                    is_rotated_list.append(True)
                else:
                    # 水平框: 转换为角点坐标
                    x_center_abs = x_center * img_w
                    y_center_abs = y_center * img_h
                    width_abs = width * img_w
                    height_abs = height * img_h

                    x_min = x_center_abs - width_abs / 2
                    y_min = y_center_abs - height_abs / 2
                    x_max = x_center_abs + width_abs / 2
                    y_max = y_center_abs + height_abs / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    is_rotated_list.append(False)

        return np.array(boxes) if boxes else np.array([]), np.array(is_rotated_list)

    @staticmethod
    def rotated_box_to_horizontal(box: np.ndarray) -> np.ndarray:
        """
        将旋转框转换为水平包围框 (x_min, y_min, x_max, y_max)

        Args:
            box: [cx, cy, w, h, angle] 旋转框参数

        Returns:
            [x_min, y_min, x_max, y_max] 水平包围框
        """
        cx, cy, w, h, angle = box
        angle_rad = np.deg2rad(angle)

        # 计算旋转矩形的四个角点
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # 半宽高
        hw = w / 2
        hh = h / 2

        # 四个角点相对于中心的偏移
        corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])

        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # 旋转角点
        rotated_corners = corners @ rotation_matrix.T

        # 平移到中心位置
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy

        # 计算水平包围框
        x_min = np.min(rotated_corners[:, 0])
        y_min = np.min(rotated_corners[:, 1])
        x_max = np.max(rotated_corners[:, 0])
        y_max = np.max(rotated_corners[:, 1])

        return np.array([x_min, y_min, x_max, y_max])

    def analyze(self, max_samples: Optional[int] = None) -> ObjectStats:
        """
        分析数据集目标分布，支持旋转框

        Args:
            max_samples: 最大分析样本数，None表示分析全部

        Returns:
            目标统计信息
        """
        stats = ObjectStats()
        size_distribution = []

        samples_to_analyze = self.samples[:max_samples] if max_samples else self.samples

        for img_path, label_path in samples_to_analyze:
            try:
                with Image.open(img_path) as img:
                    original_size = img.size  # (width, height)
            except Exception:
                continue

            boxes, is_rotated = self.parse_yolo_label(label_path, original_size)

            if len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                # 如果是旋转框，先转换为水平包围框用于尺寸分析
                if is_rotated[i]:
                    h_box = self.rotated_box_to_horizontal(box)
                    stats.rotated += 1
                else:
                    h_box = box
                    stats.horizontal += 1

                # 使用水平包围框计算面积
                width = h_box[2] - h_box[0]
                height = h_box[3] - h_box[1]
                area = width * height
                total_area = original_size[0] * original_size[1]
                relative_area = area / total_area

                # 目标尺寸分类
                if relative_area < self.SMALL_THRESHOLD:
                    stats.small += 1
                elif relative_area < self.MEDIUM_THRESHOLD:
                    stats.medium += 1
                else:
                    stats.large += 1

                stats.total += 1
                size_distribution.append(relative_area)

        # 计算统计值
        if stats.total > 0:
            stats.avg_size = float(np.mean(size_distribution))
            # 海洋环境难度评分（小目标越多，难度越高）
            stats.marine_difficulty = (
                stats.small_ratio * 0.7 + stats.medium_ratio * 0.3
            )

        return stats

    def recommend_strategy(self) -> str:
        """
        根据分析结果推荐训练策略

        Returns:
            策略名称: 'standard', 'two_stage', 'three_stage', 'synergistic'
        """
        stats = self.analyze(max_samples=1000)

        print("\n📈 数据集分析结果:")
        print(f"   总目标数: {stats.total}")
        print(f"   小目标比例: {stats.small_ratio:.2%}")
        print(f"   中目标比例: {stats.medium_ratio:.2%}")
        print(f"   大目标比例: {stats.large_ratio:.2%}")
        print(f"   旋转框数量: {stats.rotated} ({stats.rotated_ratio:.2%})")
        print(f"   水平框数量: {stats.horizontal}")
        print(f"   平均相对尺寸: {stats.avg_size:.4f}")
        print(f"   海洋检测难度: {stats.marine_difficulty:.2f}")

        if stats.small_ratio > 0.4:
            strategy = 'synergistic'
            print("\n🎯 推荐策略: 协同优化训练（数据+模型+注意力）")
        elif stats.small_ratio > 0.25:
            strategy = 'three_stage'
            print("\n🎯 推荐策略: 三阶段渐进式训练")
        elif stats.small_ratio > 0.15:
            strategy = 'two_stage'
            print("\n🎯 推荐策略: 两阶段渐进式训练")
        else:
            strategy = 'standard'
            print("\n🎯 推荐策略: 标准训练")

        return strategy


def analyze_marine_dataset(config) -> str:
    """
    分析海洋船舶数据集

    Args:
        config: 配置对象

    Returns:
        推荐策略名称
    """
    from config import Config

    train_paths = Config.get_train_paths()
    analyzer = DatasetAnalyzer(
        img_paths=train_paths["img_paths"],
        label_paths=train_paths["label_paths"]
    )

    return analyzer.recommend_strategy()


if __name__ == "__main__":
    # 独立运行测试
    from config import Config

    strategy = analyze_marine_dataset(Config)
    print(f"\n最终推荐策略: {strategy}")
