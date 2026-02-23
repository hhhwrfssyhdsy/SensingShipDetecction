"""
边缘设备模型优化与导出模块
对模型进行轻量化优化后导出为边缘设备部署格式
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积
    减少参数量和计算量，适合边缘设备
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()

        # 深度卷积 (Depthwise)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=bias
        )

        # 逐点卷积 (Pointwise)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            0,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class GhostModule(nn.Module):
    """
    Ghost模块
    通过廉价操作生成更多特征图，减少计算量
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # 主卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )

        # 廉价操作（深度可分离卷积）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # 截取前out_channels个通道
        return out[:, :self.out_channels, :, :]


class EdgeOptimizer:
    """
    边缘设备优化器
    对模型进行轻量化优化
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def optimize_for_edge(
        self,
        use_lightweight_blocks: bool = True,
        pruning_ratio: float = 0.2
    ) -> nn.Module:
        """
        执行边缘优化

        Args:
            use_lightweight_blocks: 使用轻量化模块替换标准卷积
            pruning_ratio: 剪枝比例（简化实现，实际剪枝需要更复杂的处理）

        Returns:
            优化后的模型
        """

        # 1. 替换为轻量化模块
        if use_lightweight_blocks:
            self.model = self._replace_with_lightweight(self.model)

        # 2. 通道剪枝（简化版）
        if pruning_ratio > 0:
            self.model = self._apply_channel_pruning(self.model, pruning_ratio)

        print("   ✅ 边缘优化完成")
        return self.model

    def _replace_with_lightweight(self, model: nn.Module) -> nn.Module:
        """
        替换标准模块为轻量化模块
        遍历模型所有层，将符合条件的Conv2d替换为深度可分离卷积
        """
        replaced_count = 0

        def replace_modules(module, name=""):
            nonlocal replaced_count
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                # 替换标准Conv2d（特定条件下）
                if isinstance(child_module, nn.Conv2d):
                    # 只替换特定条件的卷积：kernel_size=3, stride=1, 且不是1x1卷积
                    if (child_module.kernel_size == (3, 3) and
                        child_module.stride == (1, 1) and
                        child_module.groups == 1 and
                        child_module.in_channels >= 64 and
                        child_module.out_channels >= 64):

                        # 创建深度可分离卷积
                        ds_conv = DepthwiseSeparableConv(
                            in_channels=child_module.in_channels,
                            out_channels=child_module.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=child_module.bias is not None
                        )

                        # 尝试迁移权重
                        try:
                            with torch.no_grad():
                                # 深度卷积权重：对输入通道取平均
                                depth_weight = child_module.weight.data.mean(dim=1, keepdim=True)
                                ds_conv.depthwise.weight.data = depth_weight

                                # 点卷积权重：使用原始权重的近似
                                ds_conv.pointwise.weight.data = child_module.weight.data.mean(
                                    dim=[2, 3], keepdim=True
                                )
                        except Exception:
                            pass  # 如果权重迁移失败，使用默认初始化

                        # 替换模块
                        setattr(module, child_name, ds_conv)
                        replaced_count += 1

                # 递归处理子模块
                replace_modules(child_module, full_name)

        replace_modules(model)
        print(f"      替换了 {replaced_count} 个卷积层为深度可分离卷积")
        return model

    def _apply_channel_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """
        应用通道剪枝 - 基于L1范数计算通道重要性

        使用L1范数（权重绝对值之和）来评估每个输出通道的重要性，
        移除重要性最低的通道。

        Args:
            model: 待剪枝的模型
            pruning_ratio: 剪枝比例 (0-1)，表示要移除的通道比例

        Returns:
            剪枝后的模型
        """
        print(f"      开始通道剪枝 (剪枝比例: {pruning_ratio*100:.1f}%)")

        pruned_layers = 0
        total_channels_before = 0
        total_channels_after = 0

        # 遍历所有模块进行剪枝
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Conv2d):
                # 只对特定层进行剪枝：输出通道数大于32且不是1x1卷积或分组卷积
                if (module.out_channels > 32 and
                    module.kernel_size != (1, 1) and
                    module.groups == 1):

                    # 计算每个输出通道的重要性 (L1范数)
                    weight = module.weight.data  # shape: [out_channels, in_channels, k, k]

                    # 计算每个输出通道的L1范数
                    # 对输入通道、高度、宽度维度求绝对值之和
                    channel_importance = torch.sum(torch.abs(weight), dim=[1, 2, 3])

                    # 确定要保留的通道数
                    num_channels = module.out_channels
                    num_keep = max(1, int(num_channels * (1 - pruning_ratio)))
                    num_prune = num_channels - num_keep

                    if num_prune <= 0:
                        continue

                    # 根据重要性排序，获取要保留的通道索引
                    _, keep_indices = torch.topk(channel_importance, num_keep, largest=True, sorted=True)
                    keep_indices = keep_indices.sort()[0]  # 排序以保持顺序

                    # 创建新的卷积层
                    new_conv = nn.Conv2d(
                        in_channels=module.in_channels,
                        out_channels=num_keep,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        bias=module.bias is not None
                    )

                    # 复制权重到新的卷积层
                    with torch.no_grad():
                        new_conv.weight.data = module.weight.data[keep_indices, :, :, :]
                        if module.bias is not None:
                            new_conv.bias.data = module.bias.data[keep_indices]

                    # 替换原模块
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]

                    if parent_name:
                        parent_module = model.get_submodule(parent_name)
                        setattr(parent_module, child_name, new_conv)
                    else:
                        setattr(model, child_name, new_conv)

                    # 更新后续层的输入通道（如果当前层后面是BN或Conv）
                    self._update_next_layer_input(model, name, keep_indices)

                    pruned_layers += 1
                    total_channels_before += num_channels
                    total_channels_after += num_keep

                    print(f"         {name}: {num_channels} -> {num_keep} 通道 "
                          f"(移除了 {num_prune} 个通道)")

        if pruned_layers > 0:
            reduction = (1 - total_channels_after / total_channels_before) * 100
            print(f"      ✅ 完成 {pruned_layers} 层的剪枝")
            print(f"         总通道数: {total_channels_before} -> {total_channels_after} "
                  f"(减少 {reduction:.1f}%)")
        else:
            print(f"      ⚠️  没有符合条件的层进行剪枝")

        return model

    def _update_next_layer_input(self, model: nn.Module, current_layer_name: str, keep_indices: torch.Tensor):
        """
        更新下一层的输入通道以匹配当前层的输出

        这是一个简化实现，实际应用中需要更复杂的图分析来正确处理层之间的连接

        Args:
            model: 模型
            current_layer_name: 当前层的名称
            keep_indices: 保留的通道索引
        """
        # 获取当前层的父模块路径
        path_parts = current_layer_name.split('.')

        # 尝试找到Sequential或类似容器中的下一层
        if len(path_parts) >= 2:
            parent_path = '.'.join(path_parts[:-1])
            current_idx = path_parts[-1]

            try:
                parent_module = model.get_submodule(parent_path)

                # 如果父模块是Sequential，尝试找到下一层
                if isinstance(parent_module, nn.Sequential):
                    # 尝试将索引转换为整数
                    try:
                        current_idx_int = int(current_idx)
                        next_idx = current_idx_int + 1

                        # 检查下一层是否存在
                        if str(next_idx) in dict(parent_module.named_children()):
                            next_layer = parent_module[next_idx]

                            # 如果下一层是Conv2d，需要调整其输入通道
                            if isinstance(next_layer, nn.Conv2d):
                                self._adjust_conv_input_channels(
                                    parent_module, str(next_idx), next_layer, keep_indices
                                )
                    except ValueError:
                        pass  # 索引不是整数，跳过

            except AttributeError:
                pass  # 无法获取父模块，跳过

    def _adjust_conv_input_channels(
        self,
        parent_module: nn.Module,
        layer_name: str,
        conv_layer: nn.Conv2d,
        keep_indices: torch.Tensor
    ):
        """
        调整卷积层的输入通道

        Args:
            parent_module: 父模块
            layer_name: 层名称
            conv_layer: 卷积层
            keep_indices: 保留的输入通道索引
        """
        num_out = conv_layer.out_channels
        num_in_new = len(keep_indices)

        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=num_in_new,
            out_channels=num_out,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups if conv_layer.groups == 1 else num_in_new,  # 深度可分离卷积处理
            bias=conv_layer.bias is not None
        )

        # 复制权重：只保留需要的输入通道
        with torch.no_grad():
            new_conv.weight.data = conv_layer.weight.data[:, keep_indices, :, :]
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data.clone()

        # 替换层
        setattr(parent_module, layer_name, new_conv)

    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        opset_version: int = 11
    ) -> str:
        """
        导出模型为ONNX格式

        Args:
            output_path: 输出文件路径
            input_shape: 输入张量形状
            opset_version: ONNX算子集版本

        Returns:
            导出的文件路径
        """
        print(f"\n📤 导出ONNX模型: {output_path}")

        self.model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # 验证导出成功
        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   ✅ ONNX导出完成: {size_mb:.2f} MB")
        else:
            print(f"   ❌ ONNX导出失败")

        return output_path

    def export_to_torchscript(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640)
    ) -> str:
        """
        导出模型为TorchScript格式

        Args:
            output_path: 输出文件路径
            input_shape: 输入张量形状

        Returns:
            导出的文件路径
        """
        print(f"\n📤 导出TorchScript模型: {output_path}")

        self.model.eval()
        dummy_input = torch.randn(input_shape)

        # 使用tracing导出
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(output_path)

        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   ✅ TorchScript导出完成: {size_mb:.2f} MB")
        else:
            print(f"   ❌ TorchScript导出失败")

        return output_path


def optimize_and_export_model(
    model_path: str,
    output_dir: str,
    model_name: str = "model",
    use_lightweight: bool = True,
    pruning_ratio: float = 0.2
) -> Dict[str, str]:
    """
    优化模型并导出用于边缘设备部署

    Args:
        model_path: 训练好的模型路径 (.pt文件)
        output_dir: 输出目录
        model_name: 模型名称前缀
        use_lightweight: 是否使用轻量化模块
        pruning_ratio: 剪枝比例

    Returns:
        导出的文件路径字典
    """
    from ultralytics import YOLO

    # 加载模型
    model = YOLO(model_path)

    # 创建优化器
    optimizer = EdgeOptimizer(model.model)

    # 执行优化
    optimized_model = optimizer.optimize_for_edge(
        use_lightweight_blocks=use_lightweight,
        pruning_ratio=pruning_ratio
    )

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported_files = {}

    # 导出ONNX
    onnx_path = output_path / f"{model_name}.onnx"
    optimizer.export_to_onnx(str(onnx_path))
    exported_files['onnx'] = str(onnx_path)

    # 导出TorchScript
    torchscript_path = output_path / f"{model_name}.torchscript"
    optimizer.export_to_torchscript(str(torchscript_path))
    exported_files['torchscript'] = str(torchscript_path)


    return exported_files


def export_comparison_models(
    baseline_model_path: str,
    improved_model_path: str,
    output_dir: str = "edge_deployment",
    use_optimization: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    导出Baseline和改进模型用于边缘设备对比

    Args:
        baseline_model_path: Baseline模型路径
        improved_model_path: 改进模型路径
        output_dir: 输出目录
        use_optimization: 是否应用边缘优化

    Returns:
        两个模型的导出文件路径
    """
    results = {}

    # 导出Baseline模型
    results['baseline'] = optimize_and_export_model(
        baseline_model_path,
        f"{output_dir}/baseline",
        "baseline_yolo11s",
        use_lightweight=use_optimization,
        pruning_ratio=0.2
    )

    # 导出改进模型

    results['improved'] = optimize_and_export_model(
        improved_model_path,
        f"{output_dir}/improved",
        "improved_yolo11s",
        use_lightweight=use_optimization,
        pruning_ratio=0.2
    )



    return results
