"""
训练配置文件 - 简化版
所有参数固定，无需命令行配置
"""
import os
from pathlib import Path


class Config:
    """训练配置"""

    # 数据集路径（使用环境变量或默认值）
    SSDD_TRAIN_INSHORE_IMG = Path(os.getenv("SSDD_TRAIN_INSHORE_IMG", "D:/DataSet/SSDD/train_inshore/images"))
    SSDD_TRAIN_OFFSHORE_IMG = Path(os.getenv("SSDD_TRAIN_OFFSHORE_IMG", "D:/DataSet/SSDD/train_offshore/images"))
    SSDD_TEST_INSHORE_IMG = Path(os.getenv("SSDD_TEST_INSHORE_IMG", "D:/DataSet/SSDD/test_inshore/images"))
    SSDD_TEST_OFFSHORE_IMG = Path(os.getenv("SSDD_TEST_OFFSHORE_IMG", "D:/DataSet/SSDD/test_offshore/images"))
    SEASHIP_IMG = Path(os.getenv("SEASHIP_IMG", "D:/DataSet/SeaShips/images"))

    SSDD_TRAIN_LABEL = Path(os.getenv("SSDD_TRAIN_LABEL", "D:/DataSet/SSDD/train_inshore/labels"))
    SSDD_TEST_INSHORE_LABEL = Path(os.getenv("SSDD_TEST_INSHORE_LABEL", "D:/DataSet/SSDD/test_inshore/labels"))
    SSDD_TEST_OFFSHORE_LABEL = Path(os.getenv("SSDD_TEST_OFFSHORE_LABEL", "D:/DataSet/SSDD/test_offshore/labels"))
    SEASHIP_LABEL = Path(os.getenv("SEASHIP_LABEL", "D:/DataSet/SeaShips/labels"))

    # 训练参数
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

    # 类别配置
    SSDD_CLASSES = {"ship": 0}
    SEASHIP_CLASSES = {
        "passenger ship": 0,
        "container ship": 1,
        "ore carrier": 2,
        "general cargo ship": 3,
        "fishing boat": 4,
    }

    # 统一类别映射
    UNIFIED_CLASSES = {
        "ship": 0,
    }

    # ==================== 边缘计算优化配置 ====================

    # 边缘设备优化开关
    EDGE_OPTIMIZATION = {
        "enabled": True,  # 启用边缘优化
        "use_lightweight_blocks": True,  # 使用轻量化模块
        "use_pruning": True,  # 启用模型剪枝
        "use_quantization": True,  # 启用INT8量化
        "pruning_ratio": 0.2,  # 剪枝比例 (0-1)
        "quantization_bits": 8,  # 量化位数
    }

    # 边缘设备目标平台
    EDGE_TARGET_PLATFORMS = ["cpu", "gpu", "jetson", "raspberry_pi", "openvino"]

    # 边缘优化训练配置
    EDGE_TRAINING_CONFIG = {
        "epochs": 100,  # 边缘模型训练轮数
        "batch_size": 4,  # 较小的批次大小
        "learning_rate": 0.001,  # 较低的学习率
        "weight_decay": 0.0005,
        "label_smoothing": 0.1,
        "dropout": 0.1,  # 添加dropout防止过拟合
    }

    # 模型压缩目标
    EDGE_COMPRESSION_TARGETS = {
        "max_model_size_mb": 50,  # 最大模型大小 (MB)
        "min_fps": 15,  # 最低FPS要求
        "target_latency_ms": 66,  # 目标延迟 (ms, 约15FPS)
    }

    # ONNX导出配置
    ONNX_EXPORT_CONFIG = {
        "opset_version": 11,
        "dynamic_axes": True,
        "simplify": True,  # 使用onnx-simplifier简化模型
    }

    # TensorRT配置
    TENSORRT_CONFIG = {
        "fp16_mode": True,  # 启用FP16
        "max_batch_size": 1,
        "max_workspace_size": 1 << 30,  # 1GB工作空间
    }

    @classmethod
    def get_output_dir(cls, subdir: str = "") -> Path:
        """获取输出目录"""
        path = cls.OUTPUT_ROOT / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def validate_paths(cls) -> bool:
        """验证必要路径是否存在"""
        required_paths = [
            cls.SSDD_TRAIN_INSHORE_IMG,
            cls.SSDD_TRAIN_LABEL,
        ]

        missing = []
        for path in required_paths:
            if not path.exists():
                missing.append(str(path))

        if missing:
            print("⚠️  缺失的路径:")
            for p in missing:
                print(f"   - {p}")
            return False

        return True
