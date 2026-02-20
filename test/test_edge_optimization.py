"""
边缘计算优化模块单元测试
"""
import unittest
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.edge_optimization import (
    DepthwiseSeparableConv,
    GhostModule,
    LightweightBottleneck,
    ModelPruner,
    ModelQuantizer,
    EdgeOptimizer,
)


class TestDepthwiseSeparableConv(unittest.TestCase):
    """测试深度可分离卷积"""

    def test_initialization(self):
        """测试初始化"""
        conv = DepthwiseSeparableConv(64, 128, 3, 1, 1)
        self.assertIsNotNone(conv)
        self.assertIsNotNone(conv.depthwise)
        self.assertIsNotNone(conv.pointwise)

    def test_forward_pass(self):
        """测试前向传播"""
        conv = DepthwiseSeparableConv(64, 128, 3, 1, 1)
        x = torch.randn(2, 64, 32, 32)
        out = conv(x)
        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_parameter_reduction(self):
        """测试参数量减少"""
        # 标准卷积
        standard_conv = nn.Conv2d(64, 128, 3, 1, 1)
        standard_params = sum(p.numel() for p in standard_conv.parameters())

        # 深度可分离卷积
        ds_conv = DepthwiseSeparableConv(64, 128, 3, 1, 1)
        ds_params = sum(p.numel() for p in ds_conv.parameters())

        # 深度可分离卷积参数量应该更少
        self.assertLess(ds_params, standard_params)


class TestGhostModule(unittest.TestCase):
    """测试Ghost模块"""

    def test_initialization(self):
        """测试初始化"""
        ghost = GhostModule(64, 128, ratio=2)
        self.assertIsNotNone(ghost)
        self.assertIsNotNone(ghost.primary_conv)
        self.assertIsNotNone(ghost.cheap_operation)

    def test_forward_pass(self):
        """测试前向传播"""
        ghost = GhostModule(64, 128, ratio=2)
        x = torch.randn(2, 64, 32, 32)
        out = ghost(x)
        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_output_channels(self):
        """测试输出通道数"""
        in_channels = 64
        out_channels = 128
        ghost = GhostModule(in_channels, out_channels)
        x = torch.randn(1, in_channels, 16, 16)
        out = ghost(x)
        self.assertEqual(out.shape[1], out_channels)


class TestLightweightBottleneck(unittest.TestCase):
    """测试轻量化Bottleneck"""

    def test_initialization(self):
        """测试初始化"""
        bottleneck = LightweightBottleneck(64, 64, shortcut=True)
        self.assertIsNotNone(bottleneck)

    def test_forward_with_shortcut(self):
        """测试带shortcut的前向传播"""
        bottleneck = LightweightBottleneck(64, 64, shortcut=True)
        x = torch.randn(2, 64, 32, 32)
        out = bottleneck(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_without_shortcut(self):
        """测试不带shortcut的前向传播"""
        bottleneck = LightweightBottleneck(64, 128, shortcut=False)
        x = torch.randn(2, 64, 32, 32)
        out = bottleneck(x)
        self.assertEqual(out.shape, (2, 128, 32, 32))


class TestModelPruner(unittest.TestCase):
    """测试模型剪枝"""

    def test_initialization(self):
        """测试初始化"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
        )
        pruner = ModelPruner(model, pruning_ratio=0.3)
        self.assertIsNotNone(pruner)
        self.assertEqual(pruner.pruning_ratio, 0.3)

    def test_importance_calculation(self):
        """测试重要性计算"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
        )
        pruner = ModelPruner(model, pruning_ratio=0.3)
        pruner.calculate_importance()

        # 应该为每个卷积层计算重要性
        self.assertGreater(len(pruner.importance_scores), 0)


class TestEdgeOptimizer(unittest.TestCase):
    """测试边缘优化器"""

    def test_initialization(self):
        """测试初始化"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
        )
        optimizer = EdgeOptimizer(model)
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.model)

    def test_optimization_history(self):
        """测试优化历史记录"""
        model = nn.Sequential(nn.Conv2d(3, 64, 3))
        optimizer = EdgeOptimizer(model)

        optimizer._log_optimization("test", "completed")
        self.assertIn("test", optimizer.optimization_history)
        self.assertEqual(optimizer.optimization_history["test"]["status"], "completed")

    def test_benchmark(self):
        """测试基准测试"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        optimizer = EdgeOptimizer(model)

        results = optimizer.benchmark(input_shape=(1, 3, 64, 64), num_runs=5)

        self.assertIn("avg_inference_time_ms", results)
        self.assertIn("fps", results)
        self.assertIn("model_size_mb", results)
        self.assertIn("num_parameters", results)

        self.assertGreater(results["fps"], 0)
        self.assertGreater(results["num_parameters"], 0)


class TestEdgeConfig(unittest.TestCase):
    """测试边缘计算配置"""

    def test_edge_optimization_enabled(self):
        """测试边缘优化开关"""
        from Config.config import Config
        self.assertIn("enabled", Config.EDGE_OPTIMIZATION)

    def test_edge_optimization_options(self):
        """测试边缘优化选项"""
        from Config.config import Config

        edge_config = Config.EDGE_OPTIMIZATION
        self.assertIn("use_lightweight_blocks", edge_config)
        self.assertIn("use_pruning", edge_config)
        self.assertIn("use_quantization", edge_config)
        self.assertIn("pruning_ratio", edge_config)
        self.assertIn("quantization_bits", edge_config)

    def test_pruning_ratio_range(self):
        """测试剪枝比例范围"""
        from Config.config import Config

        ratio = Config.EDGE_OPTIMIZATION["pruning_ratio"]
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_edge_training_config(self):
        """测试边缘训练配置"""
        from Config.config import Config

        edge_training = Config.EDGE_TRAINING_CONFIG
        self.assertIn("epochs", edge_training)
        self.assertIn("batch_size", edge_training)
        self.assertIn("learning_rate", edge_training)
        self.assertIn("weight_decay", edge_training)

    def test_compression_targets(self):
        """测试压缩目标"""
        from Config.config import Config

        targets = Config.EDGE_COMPRESSION_TARGETS
        self.assertIn("max_model_size_mb", targets)
        self.assertIn("min_fps", targets)
        self.assertIn("target_latency_ms", targets)

        self.assertGreater(targets["max_model_size_mb"], 0)
        self.assertGreater(targets["min_fps"], 0)
        self.assertGreater(targets["target_latency_ms"], 0)

    def test_onnx_export_config(self):
        """测试ONNX导出配置"""
        from Config.config import Config

        onnx_config = Config.ONNX_EXPORT_CONFIG
        self.assertIn("opset_version", onnx_config)
        self.assertIn("dynamic_axes", onnx_config)
        self.assertIn("simplify", onnx_config)

    def test_tensorrt_config(self):
        """测试TensorRT配置"""
        from Config.config import Config

        trt_config = Config.TENSORRT_CONFIG
        self.assertIn("fp16_mode", trt_config)
        self.assertIn("max_batch_size", trt_config)
        self.assertIn("max_workspace_size", trt_config)


class TestEdgeTrainingIntegration(unittest.TestCase):
    """测试边缘训练集成"""

    def test_trainer_has_edge_method(self):
        """测试训练器是否有边缘训练方法"""
        from trainer import ShipDetectionTrainer

        trainer = ShipDetectionTrainer()
        self.assertTrue(hasattr(trainer, 'train_edge_optimized'))

    def test_edge_training_config_consistency(self):
        """测试边缘训练配置一致性"""
        from Config.config import Config
        from trainer import ShipDetectionTrainer

        edge_config = Config.EDGE_OPTIMIZATION
        edge_training = Config.EDGE_TRAINING_CONFIG

        # 验证配置值合理
        self.assertGreater(edge_training["epochs"], 0)
        self.assertGreater(edge_training["batch_size"], 0)
        self.assertGreater(edge_training["learning_rate"], 0)

        # 验证优化开关
        self.assertIsInstance(edge_config["enabled"], bool)
        self.assertIsInstance(edge_config["use_lightweight_blocks"], bool)
        self.assertIsInstance(edge_config["use_pruning"], bool)
        self.assertIsInstance(edge_config["use_quantization"], bool)


if __name__ == '__main__':
    unittest.main()
