"""
数据集分析模块 
"""
from pathlib import Path
from typing import Dict, List
import json

from Config.config import Config


class DatasetAnalyzer:
    """数据集分析器"""

    def __init__(self):
        self.stats = {
            "total_images": 0,
            "total_labels": 0,
            "total_objects": 0,
            "classes": {},
            "splits": {}
        }

    def analyze(self) -> Dict:
        """分析数据集"""
        print("\n" + "=" * 60)
        print("📊 分析数据集")
        print("=" * 60)
        print(f"   数据集路径: {Config.DATASET_ROOT}")

        # 分析训练集
        if Config.TRAIN_IMG_DIR.exists():
            self._analyze_split("train", Config.TRAIN_IMG_DIR, Config.TRAIN_LABEL_DIR)

        # 分析验证集
        if Config.VAL_IMG_DIR.exists():
            self._analyze_split("val", Config.VAL_IMG_DIR, Config.VAL_LABEL_DIR)

        # 分析测试集
        if Config.TEST_IMG_DIR.exists():
            self._analyze_split("test", Config.TEST_IMG_DIR, Config.TEST_LABEL_DIR)

        print(f"\n   总图片数: {self.stats['total_images']}")
        print(f"   总标签数: {self.stats['total_labels']}")
        print(f"   总目标数: {self.stats['total_objects']}")

        return self.stats

    def _analyze_split(self, name: str, img_dir: Path, label_dir: Path):
        """分析数据集划分（train/val/test）"""
        if not img_dir.exists() or not label_dir.exists():
            return

        # 支持多种图片格式
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            img_files.extend(list(img_dir.glob(ext)))

        label_files = list(label_dir.glob("*.txt"))

        # 统计目标数
        obj_count = 0
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    obj_count += len(f.readlines())
            except Exception:
                pass

        self.stats["splits"][name] = {
            "images": len(img_files),
            "labels": len(label_files),
            "objects": obj_count
        }

        self.stats["total_images"] += len(img_files)
        self.stats["total_labels"] += len(label_files)
        self.stats["total_objects"] += obj_count

        print(f"   {name}: {len(img_files)} 图片, {obj_count} 目标")

    def save_report(self, output_path: Path = None):
        """保存分析报告"""
        if output_path is None:
            output_path = Config.OUTPUT_ROOT / "reports" / "dataset_analysis.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"\n✅ 分析报告已保存: {output_path}")


def analyze_dataset():
    """分析数据集的便捷函数"""
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze()
    analyzer.save_report()
    return stats


if __name__ == "__main__":
    analyze_dataset()
