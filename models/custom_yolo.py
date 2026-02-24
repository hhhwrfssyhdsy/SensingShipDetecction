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
    注意力Hook管理器
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
        # 获取注意力类
        attention_map = {
            'se': SEAttention,
            'cbam': CBAM,
            'marine': MarineContextAttention,
        }

        if attention_type not in attention_map:
            print(f"   ⚠️  未知注意力类型: {attention_type}")
            return False

        attention_class = attention_map[attention_type]

        # 清理旧hook
        self.clear_hooks()

        # 尝试不同的模型结构访问方式
        layers = None
        model_type = None

        # 方式1: model.model.model (Ultralytics YOLO标准结构)
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            layers = model.model.model
            model_type = "model.model.model"
        # 方式2: model.model (直接模型访问)
        elif hasattr(model, 'model'):
            layers = model.model
            model_type = "model.model"

        if layers is None:
            print("   ⚠️  无法访问模型层结构")
            return False

        print(f"   模型结构类型: {model_type}, 总层数: {len(layers)}")

        # 如果没有指定目标层，自动检测特征层
        if target_layers is None:
            target_layers = self._detect_feature_layers(layers, attention_class)

        if not target_layers:
            print("   ⚠️  未找到合适的目标层")
            return False

        print(f"   将在 {len(target_layers)} 个特征层应用 {attention_type.upper()} 注意力")

        # 为每个目标层注册hook
        applied_count = 0
        for layer_idx, (name, channels) in target_layers.items():
            if layer_idx >= len(layers):
                print(f"   ⚠️  层索引 {layer_idx} 超出范围 (总层数: {len(layers)})")
                continue

            try:
                # 创建注意力模块
                attention = attention_class(channels)
                self.attention_modules[name] = attention

                # 获取目标层
                target_layer = layers[layer_idx]

                # 确保目标层是nn.Module
                if not isinstance(target_layer, nn.Module):
                    print(f"   ⚠️  层 {name} 不是nn.Module类型")
                    continue

                # 定义hook函数 - 使用闭包捕获attention模块
                # 注意: hook函数签名必须是 (module, input, output)
                # input是tuple of tensors, output是tensor
                def create_hook(attn):
                    def hook(module, inp, out):
                        # inp是tuple，我们不需要修改输入
                        # out是tensor，我们应用注意力并返回
                        return attn(out)
                    return hook

                # 注册forward hook
                hook_handle = target_layer.register_forward_hook(create_hook(attention))
                self.hooks.append(hook_handle)
                applied_count += 1
                print(f"   ✓ 已在层 {name} (索引 {layer_idx}, 通道 {channels}) 注册Hook")

            except Exception as e:
                print(f"   ⚠️  无法在层 {name} 应用注意力: {e}")
                import traceback
                traceback.print_exc()

        if applied_count > 0:
            print(f"   ✅ 成功在 {applied_count}/{len(target_layers)} 个层应用注意力")
            return True
        else:
            print(f"   ❌ 未能成功应用注意力机制")
            return False

    def _detect_feature_layers(self, layers, attention_class) -> Dict[int, tuple]:
        """
        自动检测特征层

        Args:
            layers: 模型层列表
            attention_class: 注意力类

        Returns:
            目标层配置
        """
        target_layers = {}

        # 遍历层，寻找合适的特征层
        for idx, layer in enumerate(layers):
            layer_name = layer.__class__.__name__.lower()

            # 检查是否是C2f, C3, SPPF等特征提取模块
            if any(x in layer_name for x in ['c2f', 'c3', 'sppf', 'bottleneck']):
                # 尝试获取输出通道数
                channels = None
                for m in reversed(list(layer.modules())):
                    if isinstance(m, nn.Conv2d):
                        channels = m.out_channels
                        break

                if channels:
                    target_layers[idx] = (f'{layer_name}_{idx}', channels)

        # 如果找到的层太多，只选择最后3个（通常是P3, P4, P5）
        if len(target_layers) > 3:
            # 按索引排序，取最后3个
            sorted_layers = sorted(target_layers.items(), key=lambda x: x[0])
            target_layers = dict(sorted_layers[-3:])

        # 如果没有找到，使用YOLO11s的默认配置
        if not target_layers:
            print("   使用默认层配置 (YOLO11s)")
            # YOLO11s中P3, P4, P5通常在索引 8, 12, 16 附近
            # 但YOLO11结构可能不同，尝试常见位置
            default_configs = [
                ([8, 12, 16], [128, 256, 512]),  # YOLOv8风格
                ([10, 14, 18], [128, 256, 512]), # 备选
                ([7, 11, 15], [128, 256, 512]),  # 备选
            ]

            for indices, channels in default_configs:
                if all(idx < len(layers) for idx in indices):
                    for i, (idx, ch) in enumerate(zip(indices, channels)):
                        target_layers[idx] = (f'P{i+3}', ch)
                    break

        return target_layers

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

    def apply_attention(self, attention_type: str = 'se') -> bool:
        """
        应用注意力机制

        Args:
            attention_type: 注意力类型

        Returns:
            是否成功应用
        """
        self.current_attention = attention_type
        success = self.hook_manager.apply_attention(self.model, attention_type)
        return success

    def train(self, **kwargs):
        """
        训练模型，确保注意力机制生效
        """
        # 如果设置了注意力，确保在训练前应用
        if self.current_attention:
            success = self.hook_manager.apply_attention(self.model, self.current_attention)
            if not success:
                print("⚠️  警告: 注意力机制未能成功应用，将使用基础模型训练")

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
) -> Tuple[YOLO, bool]:
    """
    创建改进模型

    Args:
        attention_type: 注意力类型 ('se', 'cbam', 'marine')
        apply_hook: 是否立即应用Hook

    Returns:
        (改进的YOLO模型, 是否成功应用注意力)
    """
    # 创建改进版YOLO
    model = ImprovedYOLO("yolo11s-obb.pt")

    # 应用注意力
    success = True
    if apply_hook:
        success = model.apply_attention(attention_type)
        if success:
            print(f"   ✅ 成功应用 {attention_type.upper()} 注意力机制")
        else:
            print(f"   ⚠️  未能应用 {attention_type.upper()} 注意力机制，将使用基础模型")

    return model, success


def apply_attention_to_model(model: YOLO, attention_type: str) -> Tuple[YOLO, bool]:
    """
    为已有模型应用注意力机制

    Args:
        model: 已有YOLO模型
        attention_type: 注意力类型

    Returns:
        (应用注意力后的模型, 是否成功应用)
    """
    # 创建hook管理器
    hook_manager = AttentionHookManager()

    # 应用注意力
    success = False
    if hasattr(model, 'model'):
        success = hook_manager.apply_attention(model.model, attention_type)

        # 将管理器附加到模型，防止被回收
        model._hook_manager = hook_manager

        if success:
            print(f"   ✅ 成功为加载的模型应用 {attention_type.upper()} 注意力")
        else:
            print(f"   ⚠️  未能为加载的模型应用注意力机制")

    return model, success


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
