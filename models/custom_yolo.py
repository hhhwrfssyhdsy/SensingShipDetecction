"""
自定义YOLO模型
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict
from ultralytics import YOLO

from .attention_modules import SEAttention, CBAM, MarineContextAttention


class AttentionHookManager:
    """
    注意力Hook管理器 (方案2: Forward 
    无需修改模型结构，在forward时动态应用注意力
    """

    def __init__(self):
        self.hooks = []
        self.attention_modules = {}

    def apply_attention(self, model: nn.Module, attention_type: str,
                        target_layers: Dict[int, tuple] = None):
        """
        为模型应用注意力机制

        Args:
            model: YOLO模型
            attention_type: 注意力类型 ('se', 'cbam', 'marine')
            target_layers: 目标层配置 {层索引: (层名称, 通道数)}
        """
        # 默认在P3, P4, P5后添加注意力
        if target_layers is None:
            target_layers = {
                8: ('P3', 128),
                12: ('P4', 256),
                16: ('P5', 512)
            }

        # 获取注意力类
        attention_map = {
            'se': SEAttention,
            'cbam': CBAM,
            'marine': MarineContextAttention,
        }

        if attention_type not in attention_map:
            print(f"   ⚠️  未知注意力类型: {attention_type}")
            return

        attention_class = attention_map[attention_type]

        # 获取模型的层
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            layers = model.model.model
        else:
            print("   ⚠️  模型结构不支持Hook")
            return

        # 清理旧hook
        self.clear_hooks()

        # 为每个目标层注册hook
        for layer_idx, (name, channels) in target_layers.items():
            if layer_idx >= len(layers):
                continue

            # 创建注意力模块
            attention = attention_class(channels)
            self.attention_modules[name] = attention

            # 获取目标层
            target_layer = layers[layer_idx]

            # 定义hook函数
            def make_hook(attn_module):
                def hook(module, input, output):
                    return attn_module(output)
                return hook

            # 注册hook
            hook = target_layer.register_forward_hook(make_hook(attention))
            self.hooks.append(hook)

    def clear_hooks(self):
        """清理所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_modules.clear()


class ImprovedYOLO(YOLO):
    """
    改进版YOLO (方案3: 继承)
    继承YOLO类，添加注意力机制支持
    """

    def __init__(self, model='yolo11s-obb.pt', task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)
        self.hook_manager = AttentionHookManager()
        self.current_attention = None

    def apply_attention(self, attention_type: str = 'se'):
        """
        应用注意力机制

        Args:
            attention_type: 注意力类型
        """
        self.current_attention = attention_type
        self.hook_manager.apply_attention(self.model, attention_type)
        return self

    def train(self, **kwargs):
        """
        训练模型，确保注意力机制生效
        """
        # 如果设置了注意力，确保在训练前应用
        if self.current_attention:
            self.hook_manager.apply_attention(self.model, self.current_attention)

        # 调用父类训练
        return super().train(**kwargs)


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
    apply_hook: bool = True
) -> YOLO:
    """
    创建改进模型

    Args:
        attention_type: 注意力类型 ('se', 'cbam', 'marine')
        apply_hook: 是否立即应用Hook

    Returns:
        改进的YOLO模型
    """
    # 创建改进版YOLO
    model = ImprovedYOLO("yolo11s-obb.pt")

    # 应用注意力
    if apply_hook:
        model.apply_attention(attention_type)

    return model


def apply_attention_to_model(model: YOLO, attention_type: str) -> YOLO:
    """
    为已有模型应用注意力机制

    Args:
        model: 已有YOLO模型
        attention_type: 注意力类型

    Returns:
        应用注意力后的模型
    """
    # 创建hook管理器
    hook_manager = AttentionHookManager()

    # 应用注意力
    if hasattr(model, 'model'):
        hook_manager.apply_attention(model.model, attention_type)

        # 将管理器附加到模型，防止被回收
        model._hook_manager = hook_manager
    return model


# 兼容性函数，保持原有接口
def modify_yolo_for_training(trainer, attention_type: str = 'se'):
    """
    在训练时修改模型（用于Ultralytics回调）

    Args:
        trainer: Ultralytics trainer对象
        attention_type: 注意力类型
    """
    if hasattr(trainer, 'model') and trainer.model is not None:
        hook_manager = AttentionHookManager()
        hook_manager.apply_attention(trainer.model, attention_type)
        trainer._hook_manager = hook_manager


if __name__ == "__main__":
    # 测试
    print("测试改进模型创建...")

    # 测试基线模型
    baseline = create_baseline_model()
    print(f"   基线模型: {type(baseline)}")

    # 测试改进模型
    improved = create_improved_model("se")
    print(f"   改进模型: {type(improved)}")
    print(f"   当前注意力: {improved.current_attention}")

    print("测试完成")
