"""
训练配置文件
"""
import os
from pathlib import Path


class Config:
    """训练配置"""

    # ==================== 数据集路径配置 ====================
    # 获取项目根目录（config.py所在目录的父目录）
    PROJECT_ROOT = Path(__file__).parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset"

    # 训练/验证/测试集路径（统一格式）
    TRAIN_IMG_DIR = DATASET_ROOT / "train" / "images"
    TRAIN_LABEL_DIR = DATASET_ROOT / "train" / "labels"
    VAL_IMG_DIR = DATASET_ROOT / "val" / "images"
    VAL_LABEL_DIR = DATASET_ROOT / "val" / "labels"
    TEST_IMG_DIR = DATASET_ROOT / "test" / "images"
    TEST_LABEL_DIR = DATASET_ROOT / "test" / "labels"

    # ==================== 训练参数 ====================
    TARGET_SIZE = int(os.getenv("TARGET_SIZE", "640"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    DEVICE = os.getenv("DEVICE", "0")

    # 输出路径
    OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "D:/ShipDetection_improved"))

    # 模型配置
    BASELINE_EPOCHS = 130
    BASELINE_MODEL = "yolo11s-obb.pt"

    # 三阶段渐进训练配置
    PROGRESSIVE_STAGES = [
        {
            "name": "stage1_foundation",
            "epochs": 60,
            "attention": "se",
            "description": "基础训练 - SE注意力",
            "lr0": 0.01,
            "lrf": 0.01,
            "mosaic": 0.2,
            "mixup": 0.02,
            "box": 5.0,
        },
        {
            "name": "stage2_enhancement",
            "epochs": 40,
            "attention": "cbam",
            "description": "增强训练 - CBAM注意力",
            "lr0": 0.005,
            "lrf": 0.01,
            "mosaic": 0.5,
            "mixup": 0.1,
            "box": 6.5,
        },
        {
            "name": "stage3_refinement",
            "epochs": 30,
            "attention": "marine",
            "description": "精细优化 - MarineContext注意力",
            "lr0": 0.002,
            "lrf": 0.01,
            "mosaic": 0.8,
            "mixup": 0.2,
            "copy_paste": 0.15,
            "box": 8.0,
        },
    ]

    # 边缘优化配置
    EDGE_OPTIMIZATION = {
        "enabled": True,
        "use_lightweight_blocks": True,
        "use_pruning": True,
        "use_quantization": True,
        "pruning_ratio": 0.2,
        "quantization_bits": 8,
    }

    # 边缘设备目标平台
    EDGE_TARGET_PLATFORMS = ["cpu", "gpu", "jetson", "raspberry_pi", "openvino"]

    # 边缘优化训练配置
    EDGE_TRAINING_CONFIG = {
        "epochs": 100,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "label_smoothing": 0.1,
        "dropout": 0.1,
    }
