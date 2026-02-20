"""
数据集分析模块 - 简化版
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
            "datasets": {}
        }

    def analyze(self) -> Dict:
        """分析所有数据集"""
        print("\n" + "=" * 60)
        print("📊 分析数据集")
        print("=" * 60)

        # 分析SSDD
        if Config.SSDD_TRAIN_INSHORE_IMG.exists():
            self._analyze_dataset("SSDD_train", Config.SSDD_TRAIN_INSHORE_IMG, Config.SSDD_TRAIN_LABEL)

        # 分析SeaShips
        if Config.SEASHIP_IMG.exists():
            self._analyze_dataset("SeaShips", Config.SEASHIP_IMG, Config.SEASHIP_LABEL)

        print(f"\n   总图片数: {self.stats['total_images']}")
        print(f"   总标签数: {self.stats['total_labels']}")
        print(f"   总目标数: {self.stats['total_objects']}")

        return self.stats

    def _analyze_dataset(self, name: str, img_dir: Path, label_dir: Path):
        """分析单个数据集"""
        img_files = list(img_dir.glob("*.jpg"))
        label_files = list(label_dir.glob("*.txt"))

        obj_count = 0
        for label_file in label_files:
            with open(label_file, 'r') as f:
                obj_count += len(f.readlines())

        self.stats["datasets"][name] = {
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
            output_path = Config.get_output_dir("reports") / "dataset_analysis.json"

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
