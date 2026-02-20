"""
船舶检测训练器
支持多种训练策略：标准训练、渐进式训练、协同优化训练
统一使用旋转框(OBB)格式
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch
from ultralytics import YOLO

from Config.config import Config, TRAINING_STRATEGIES, PROGRESSIVE_STAGES


class ShipDetectionTrainer:
    """船舶检测训练器，统一使用旋转框(OBB)格式"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化训练器

        Args:
            model_path: 预训练模型路径，默认使用配置中的路径
        """
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.data_config_path = self._create_dataset_config()
        self.device = self._get_device()

    def _get_device(self) -> str:
        """获取训练设备"""
        if Config.DEVICE != "auto":
            return Config.DEVICE
        return "0" if torch.cuda.is_available() else "cpu"

    def _create_dataset_config(self) -> str:
        """创建YOLO数据集配置文件"""
        data_config = {
            "path": Config.OUTPUT_ROOT,
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["ship"]
        }

        config_dir = Path("./Config")
        config_dir.mkdir(exist_ok=True)

        yaml_path = config_dir / "ship_detection.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True)

        return str(yaml_path)

    def _get_model_path(self) -> str:
        """
        获取OBB模型路径
        将标准模型路径自动转换为OBB版本
        """
        model_path = self.model_path

        # 如果已经是OBB模型，直接返回
        if "-obb" in model_path:
            return model_path

        # 将标准模型替换为OBB版本
        if "yolo11s.pt" in model_path:
            return model_path.replace("yolo11s.pt", "yolo11s-obb.pt")
        elif "yolo11n.pt" in model_path:
            return model_path.replace("yolo11n.pt", "yolo11n-obb.pt")
        elif "yolo11m.pt" in model_path:
            return model_path.replace("yolo11m.pt", "yolo11m-obb.pt")
        elif "yolo11l.pt" in model_path:
            return model_path.replace("yolo11l.pt", "yolo11l-obb.pt")
        elif "yolo11x.pt" in model_path:
            return model_path.replace("yolo11x.pt", "yolo11x-obb.pt")

        return model_path

    def _get_base_args(self) -> Dict[str, Any]:
        """获取基础训练参数，统一使用OBB任务"""
        return {
            "data": self.data_config_path,
            "task": "obb",  # 统一使用旋转框检测任务
            "imgsz": Config.TARGET_SIZE,
            "batch": Config.BATCH_SIZE,
            "device": self.device,
            "workers": Config.NUM_WORKERS,
            "save": True,
            "pretrained": True,
            "optimizer": "auto",
            "verbose": True,
        }

    def train_standard(self) -> Tuple[YOLO, Any]:
        """标准YOLOv11-s训练"""
        model_path = self._get_model_path()
        print("🎯 开始标准YOLOv11-s模型训练...")
        print("   任务类型: 旋转框检测(OBB)")

        model = YOLO(model_path)
        args = self._get_base_args()
        args.update(TRAINING_STRATEGIES["standard"])

        results = model.train(**args)
        return model, results

    def train_small_object(self) -> Tuple[YOLO, Any]:
        """小目标优化训练"""
        model_path = self._get_model_path()
        print("🎯 开始小目标优化训练...")
        print("   任务类型: 旋转框检测(OBB)")

        model = YOLO(model_path)
        args = self._get_base_args()
        args.update(TRAINING_STRATEGIES["small_object"])

        results = model.train(**args)
        return model, results

    def train_marine(self) -> Tuple[YOLO, Any]:
        """海洋环境小目标检测训练"""
        model_path = self._get_model_path()
        print("🌊 开始海洋环境小目标检测训练...")
        print("   任务类型: 旋转框检测(OBB)")

        model = YOLO(model_path)
        args = self._get_base_args()
        args.update(TRAINING_STRATEGIES["marine"])

        results = model.train(**args)
        return model, results

    def train_progressive(self, strategy: str = "two_stage") -> Tuple[YOLO, Dict]:
        """
        渐进式训练

        Args:
            strategy: 渐进策略，可选 'two_stage' 或 'three_stage'

        Returns:
            最终模型和各阶段结果
        """
        stages = PROGRESSIVE_STAGES.get(strategy)
        if not stages:
            raise ValueError(f"未知的渐进策略: {strategy}")

        print(f"🔄 开始{strategy}渐进式训练...")

        results = {}
        model = None

        # 阶段1: 基础训练
        stage1_config = stages["stage1"]
        print(f"🎯 阶段1: {stage1_config['name']} - 建立基础检测能力")
        print("   任务类型: 旋转框检测(OBB)")

        model_path = self._get_model_path()
        model = YOLO(model_path)
        args = self._get_base_args()
        args.update(stage1_config)
        results["stage1"] = model.train(**args)

        # 后续阶段
        for stage_name, stage_config in list(stages.items())[1:]:
            print(f"🔍 {stage_name}: {stage_config['name']}")

            # 加载上一阶段的最佳模型
            prev_stage_key = list(stages.keys())[list(stages.keys()).index(stage_name)-1]
            prev_stage_name = stages[prev_stage_key]['name']
            prev_model_path = f"runs/obb/{prev_stage_name}/weights/best.pt"
            if not Path(prev_model_path).exists():
                prev_model_path = f"runs/obb/{prev_stage_name}/weights/last.pt"

            model = YOLO(prev_model_path)
            args = self._get_base_args()
            args.update(stage_config)
            args["pretrained"] = False  # 从上一阶段继续训练

            results[stage_name] = model.train(**args)

        return model, results

    def train_synergistic(self) -> Tuple[YOLO, Dict]:
        """
        协同优化训练 - 数据增强 + 模型结构 + 注意力机制
        使用三阶段渐进式训练实现协同优化
        """
        print("🚀 启动协同优化训练...")
        print("📋 训练策略: 数据增强 + 模型结构 + 注意力机制 三重优化")

        return self.train_progressive("synergistic")

    def train(self, strategy: str = "auto") -> Tuple[YOLO, Any]:
        """
        根据策略执行训练

        Args:
            strategy: 训练策略，可选:
                - 'auto': 自动选择（基于数据集分析）
                - 'standard': 标准训练
                - 'small_object': 小目标优化
                - 'marine': 海洋环境优化
                - 'two_stage': 两阶段渐进
                - 'three_stage': 三阶段渐进
                - 'synergistic': 协同优化

        Returns:
            训练好的模型和结果
        """
        if strategy == "auto":
            # 自动选择策略
            from dataset_analyzer import analyze_marine_dataset
            strategy = analyze_marine_dataset(Config)

        strategy_map = {
            "standard": self.train_standard,
            "small_object": self.train_small_object,
            "marine": self.train_marine,
            "two_stage": lambda: self.train_progressive("two_stage"),
            "three_stage": lambda: self.train_progressive("three_stage"),
            "synergistic": self.train_synergistic,
        }

        if strategy not in strategy_map:
            raise ValueError(f"未知的训练策略: {strategy}")

        return strategy_map[strategy]()


def train_ship_detection(strategy: str = "auto") -> Tuple[YOLO, Any]:
    """
    便捷的训练函数

    Args:
        strategy: 训练策略

    Returns:
        训练好的模型和结果
    """
    trainer = ShipDetectionTrainer()
    return trainer.train(strategy)


if __name__ == "__main__":
    # 独立运行测试
    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "auto"
    model, results = train_ship_detection(strategy)
    print("✅ 训练完成！")
