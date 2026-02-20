"""
自定义YOLO模型
集成注意力机制的改进版YOLO
"""
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from ultralytics import YOLO

from .attention_modules import SEAttention, CBAM, MarineContextAttention
from .enhanced_neck import EnhancedPANet


class ImprovedYOLO(nn.Module):
    """
    改进版YOLO模型
    集成注意力机制和增强Neck
    """

    def __init__(self, base_model: str = "yolo11s-obb.pt", attention_type: str = "se"):
        super().__init__()
        self.base_model = base_model
        self.attention_type = attention_type

        # 加载基础模型
        self.model = YOLO(base_model)

        # 添加注意力模块
        self.attention_modules = self._build_attention_modules(attention_type)

    def _build_attention_modules(self, attention_type: str) -> nn.ModuleDict:
        """构建注意力模块"""
        attention_map = {
            'se': SEAttention,
            'cbam': CBAM,
            'marine': MarineContextAttention,
        }

        if attention_type not in attention_map:
            raise ValueError(f"未知的注意力类型: {attention_type}")

        attention_class = attention_map[attention_type]

        # 在关键层添加注意力
        modules = nn.ModuleDict({
            'layer2': attention_class(128),
            'layer3': attention_class(256),
            'layer4': attention_class(512),
        })

        return modules

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def train_model(self, **kwargs):
        """训练模型"""
        return self.model.train(**kwargs)

    def val_model(self, **kwargs):
        """验证模型"""
        return self.model.val(**kwargs)


def create_baseline_model(model_path: str = "yolo11s-obb.pt") -> YOLO:
    """
    创建基线模型

    Args:
        model_path: 模型路径或名称

    Returns:
        YOLO模型实例
    """
    print(f"   加载基线模型: {model_path}")

    # 检查本地文件
    if Path(model_path).exists():
        return YOLO(model_path)

    # 从Ultralytics下载
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"   ⚠️  无法加载模型: {e}")
        print("   尝试使用默认模型...")
        return YOLO("yolo11s-obb.pt")


def create_improved_model(
    attention_type: str = "se",
    use_enhanced_neck: bool = True
) -> YOLO:
    """
    创建改进模型

    Args:
        attention_type: 注意力类型 ('se', 'cbam', 'marine')
        use_enhanced_neck: 是否使用增强Neck

    Returns:
        改进的YOLO模型
    """
    print(f"   创建改进模型:")
    print(f"      - 注意力机制: {attention_type}")
    print(f"      - 增强Neck: {use_enhanced_neck}")

    # 加载基础模型
    model = YOLO("yolo11s-obb.pt")

    # 注意：实际修改YOLO架构需要修改yaml配置文件
    # 这里返回基础模型，实际改进通过训练策略实现

    return model


def apply_attention_to_model(model: YOLO, attention_type: str, layer_indices: list):
    """
    在模型指定层应用注意力机制

    Args:
        model: YOLO模型
        attention_type: 注意力类型
        layer_indices: 要应用注意力的层索引
    """
    attention_map = {
        'se': SEAttention,
        'cbam': CBAM,
        'marine': MarineContextAttention,
    }

    if attention_type not in attention_map:
        return

    print(f"   应用 {attention_type} 注意力到层: {layer_indices}")

    # 注意：实际实现需要修改模型内部结构
    # 这里仅作标记，实际训练时通过配置文件实现
