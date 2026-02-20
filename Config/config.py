"""
船舶检测项目配置管理
支持环境变量覆盖，便于不同环境部署
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple


class Config:
    """项目配置类"""

    # 数据集路径配置（可通过环境变量覆盖）
    SSDD_TRAIN_INSHORE_IMG = os.getenv(
        "SSDD_TRAIN_INSHORE_IMG",
        "/root/SSDD/JPEGImages_train_inshore"
    )
    SSDD_TRAIN_OFFSHORE_IMG = os.getenv(
        "SSDD_TRAIN_OFFSHORE_IMG",
        "/root/SSDD/JPEGImages_train_offshore"
    )
    SSDD_TEST_INSHORE_IMG = os.getenv(
        "SSDD_TEST_INSHORE_IMG",
        "/root/SSDD/JPEGImages_test_inshore"
    )
    SSDD_TEST_OFFSHORE_IMG = os.getenv(
        "SSDD_TEST_OFFSHORE_IMG",
        "/root/SSDD/JPEGImages_test_offshore"
    )
    SSDD_TRAIN_LABEL = os.getenv(
        "SSDD_TRAIN_LABEL",
        "/root/SSDD/train_labels"
    )
    SSDD_TEST_INSHORE_LABEL = os.getenv(
        "SSDD_TEST_INSHORE_LABEL",
        "/root/SSDD/test_inshore_labels"
    )
    SSDD_TEST_OFFSHORE_LABEL = os.getenv(
        "SSDD_TEST_OFFSHORE_LABEL",
        "/root/SSDD/test_offshore_labels"
    )

    # SeaShip数据集路径
    SEASHIP_IMG = os.getenv(
        "SEASHIP_IMG",
        "/root/SeaShips(7000)/JPEGImages"
    )
    SEASHIP_LABEL = os.getenv(
        "SEASHIP_LABEL",
        "/root/SeaShips(7000)/labels"
    )

    # 模型路径
    YOLO_MODEL_PATH = os.getenv(
        "YOLO_MODEL_PATH",
        "/root/ShipDetection_improved/Models/yolo11s.pt"
    )

    # 输出路径
    OUTPUT_ROOT = os.getenv(
        "OUTPUT_ROOT",
        "/root/ShipDetection_improved"
    )

    # 训练参数
    TARGET_SIZE: int = int(os.getenv("TARGET_SIZE", "640"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
    EPOCHS: int = int(os.getenv("EPOCHS", "100"))
    PATIENCE: int = int(os.getenv("PATIENCE", "15"))

    # 设备配置
    DEVICE: str = os.getenv("DEVICE", "auto")

    @classmethod
    def get_train_paths(cls) -> Dict[str, List[str]]:
        """获取训练数据路径"""
        return {
            "img_paths": [
                cls.SSDD_TRAIN_INSHORE_IMG,
                cls.SSDD_TRAIN_OFFSHORE_IMG,
                cls.SEASHIP_IMG
            ],
            "label_paths": [
                cls.SSDD_TRAIN_LABEL,
                cls.SSDD_TRAIN_LABEL,
                cls.SEASHIP_LABEL
            ]
        }

    @classmethod
    def get_val_paths(cls) -> Dict[str, List[str]]:
        """获取验证数据路径"""
        return {
            "img_paths": [
                cls.SSDD_TEST_INSHORE_IMG,
                cls.SSDD_TEST_OFFSHORE_IMG
            ],
            "label_paths": [
                cls.SSDD_TEST_INSHORE_LABEL,
                cls.SSDD_TEST_OFFSHORE_LABEL
            ]
        }

    @classmethod
    def validate_paths(cls) -> Tuple[bool, List[str]]:
        """验证所有路径是否存在"""
        errors = []
        paths_to_check = [
            ("SSDD_TRAIN_INSHORE_IMG", cls.SSDD_TRAIN_INSHORE_IMG),
            ("SSDD_TRAIN_OFFSHORE_IMG", cls.SSDD_TRAIN_OFFSHORE_IMG),
            ("SSDD_TEST_INSHORE_IMG", cls.SSDD_TEST_INSHORE_IMG),
            ("SSDD_TEST_OFFSHORE_IMG", cls.SSDD_TEST_OFFSHORE_IMG),
            ("SSDD_TRAIN_LABEL", cls.SSDD_TRAIN_LABEL),
            ("SSDD_TEST_INSHORE_LABEL", cls.SSDD_TEST_INSHORE_LABEL),
            ("SSDD_TEST_OFFSHORE_LABEL", cls.SSDD_TEST_OFFSHORE_LABEL),
            ("SEASHIP_IMG", cls.SEASHIP_IMG),
            ("SEASHIP_LABEL", cls.SEASHIP_LABEL),
        ]

        for name, path in paths_to_check:
            if not Path(path).exists():
                errors.append(f"{name}: {path} 不存在")

        return len(errors) == 0, errors


# 训练策略配置
TRAINING_STRATEGIES = {
    "standard": {
        "description": "标准YOLOv11-s训练",
        "epochs": 100,
        "patience": 15,
        "mosaic": 0.3,
        "mixup": 0.05,
        "copy_paste": 0.02,
        "box": 5.0,
        "cls": 1.0,
        "dfl": 1.5,
        "lr0": 0.01,
        "lrf": 0.001,
    },
    "small_object": {
        "description": "小目标优化训练",
        "epochs": 100,
        "patience": 20,
        "mosaic": 0.7,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "box": 7.5,
        "cls": 0.6,
        "dfl": 1.5,
        "lr0": 0.01,
        "lrf": 0.01,
        "anchor_t": 3.0,
        "fl_gamma": 1.5,
    },
    "marine": {
        "description": "海洋环境小目标检测",
        "epochs": 100,
        "patience": 20,
        "mosaic": 0.7,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "box": 7.5,
        "cls": 0.6,
        "dfl": 1.5,
        "lr0": 0.01,
        "lrf": 0.01,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "anchor_t": 3.0,
        "fl_gamma": 1.5,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,
    }
}

# 渐进式训练阶段配置
PROGRESSIVE_STAGES = {
    "two_stage": {
        "stage1": {
            "name": "ship_detection_stage1",
            "epochs": 80,
            "patience": 10,
            "mosaic": 0.3,
            "mixup": 0.05,
            "copy_paste": 0.02,
            "box": 5.0,
            "cls": 1.0,
            "dfl": 1.5,
            "lr0": 0.01,
            "lrf": 0.001,
        },
        "stage2": {
            "name": "ship_detection_final",
            "epochs": 50,
            "patience": 8,
            "mosaic": 0.7,
            "mixup": 0.15,
            "copy_paste": 0.1,
            "box": 7.5,
            "cls": 0.6,
            "dfl": 2.0,
            "lr0": 0.001,
            "lrf": 0.0001,
            "hsv_h": 0.01,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 8.0,
            "translate": 0.05,
            "scale": 0.3,
        }
    },
    "three_stage": {
        "stage1": {
            "name": "stage1_normal",
            "epochs": 60,
            "patience": 8,
            "mosaic": 0.2,
            "mixup": 0.02,
        },
        "stage2": {
            "name": "stage2_balanced",
            "epochs": 40,
            "patience": 6,
            "mosaic": 0.5,
            "mixup": 0.1,
            "box": 6.0,
            "cls": 0.8,
            "lr0": 0.005,
        },
        "stage3": {
            "name": "stage3_small_objects",
            "epochs": 30,
            "patience": 5,
            "mosaic": 0.8,
            "mixup": 0.2,
            "copy_paste": 0.15,
            "box": 8.0,
            "cls": 0.5,
            "lr0": 0.001,
        }
    },
    "synergistic": {
        "stage1": {
            "name": "stage1_enhanced",
            "epochs": 60,
            "patience": 8,
            "mosaic": 0.3,
            "mixup": 0.05,
            "copy_paste": 0.02,
        },
        "stage2": {
            "name": "stage2_model_enhanced",
            "epochs": 40,
            "patience": 6,
            "mosaic": 0.5,
            "mixup": 0.1,
            "lr0": 0.005,
            "box": 7.0,
            "cls": 0.7,
        },
        "stage3": {
            "name": "stage3_attention_enhanced",
            "epochs": 30,
            "patience": 5,
            "mosaic": 0.8,
            "mixup": 0.2,
            "lr0": 0.001,
            "box": 8.0,
            "cls": 0.5,
            "dfl": 2.0,
        }
    }
}
