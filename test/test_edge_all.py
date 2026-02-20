"""
边缘计算优化模块完整测试
"""
import unittest
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEdgeModules(unittest.TestCase):
    """测试边缘计算模块"""

    def test_import_edge_optimization(self):
        """测试导入边缘优化模块"""
        try:
            from models.edge_optimization import (
                DepthwiseSeparableConv,
                GhostModule,
                LightweightBottleneck,
                EdgeOptimizer,
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")

    def test_depthwise_separable_conv(self):
        """测试深度可分离卷积"""
        from models.edge_optimization import DepthwiseSeparableConv

        conv = DepthwiseSeparableConv(64, 128, 3, 1, 1)
        x = torch.randn(2, 64, 32, 32)
        out = conv(x)

        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_ghost_module(self):
        """测试Ghost模块"""
        from models.edge_optimization import GhostModule

        ghost = GhostModule(64, 128, ratio=2)
        x = torch.randn(2, 64, 32, 32)
        out = ghost(x)

        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_lightweight_bottleneck(self):
        """测试轻量化Bottleneck"""
        from models.edge_optimization import LightweightBottleneck

        bottleneck = LightweightBottleneck(64, 64, shortcut=True)
        x = torch.randn(2, 64, 32, 32)
        out = bottleneck(x)

        self.assertEqual(out.shape, x.shape)


class TestEdgeConfig(unittest.TestCase):
    """测试边缘计算配置"""

    def test_edge_optimization_config_exists(self):
        """测试边缘优化配置存在"""
        from Config.config import Config

        self.assertTrue(hasattr(Config, 'EDGE_OPTIMIZATION'))
        self.assertIn('enabled', Config.EDGE_OPTIMIZATION)

    def test_edge_training_config(self):
        """测试边缘训练配置"""
        from Config.config import Config

        self.assertTrue(hasattr(Config, 'EDGE_TRAINING_CONFIG'))
        edge_training = Config.EDGE_TRAINING_CONFIG

        self.assertIn('epochs', edge_training)
        self.assertIn('batch_size', edge_training)
        self.assertIn('learning_rate', edge_training)

        self.assertGreater(edge_training['epochs'], 0)
        self.assertGreater(edge_training['batch_size'], 0)
        self.assertGreater(edge_training['learning_rate'], 0)

    def test_edge_compression_targets(self):
        """测试压缩目标配置"""
        from Config.config import Config

        self.assertTrue(hasattr(Config, 'EDGE_COMPRESSION_TARGETS'))
        targets = Config.EDGE_COMPRESSION_TARGETS

        self.assertIn('max_model_size_mb', targets)
        self.assertIn('min_fps', targets)
        self.assertIn('target_latency_ms', targets)

    def test_onnx_export_config(self):
        """测试ONNX导出配置"""
        from Config.config import Config

        self.assertTrue(hasattr(Config, 'ONNX_EXPORT_CONFIG'))
        onnx_config = Config.ONNX_EXPORT_CONFIG

        self.assertIn('opset_version', onnx_config)
        self.assertIn('dynamic_axes', onnx_config)

    def test_tensorrt_config(self):
        """测试TensorRT配置"""
        from Config.config import Config

        self.assertTrue(hasattr(Config, 'TENSORRT_CONFIG'))
        trt_config = Config.TENSORRT_CONFIG

        self.assertIn('fp16_mode', trt_config)
        self.assertIn('max_batch_size', trt_config)


class TestEdgeTraining(unittest.TestCase):
    """测试边缘训练功能"""

    def test_trainer_has_edge_method(self):
        """测试训练器有边缘训练方法"""
        from trainer import ShipDetectionTrainer

        trainer = ShipDetectionTrainer()
        self.assertTrue(hasattr(trainer, 'train_edge_optimized'))

    def test_edge_training_config_values(self):
        """测试边缘训练配置值"""
        from Config.config import Config

        edge_config = Config.EDGE_OPTIMIZATION

        # 验证布尔值
        self.assertIsInstance(edge_config['enabled'], bool)
        self.assertIsInstance(edge_config['use_lightweight_blocks'], bool)
        self.assertIsInstance(edge_config['use_pruning'], bool)
        self.assertIsInstance(edge_config['use_quantization'], bool)

        # 验证数值范围
        self.assertGreaterEqual(edge_config['pruning_ratio'], 0.0)
        self.assertLessEqual(edge_config['pruning_ratio'], 1.0)

        self.assertIn(edge_config['quantization_bits'], [8, 16, 32])


class TestParameterReduction(unittest.TestCase):
    """测试参数量减少"""

    def test_depthwise_vs_standard(self):
        """测试深度可分离卷积vs标准卷积"""
        from models.edge_optimization import DepthwiseSeparableConv

        # 标准卷积
        standard = nn.Conv2d(64, 128, 3, 1, 1)
        standard_params = sum(p.numel() for p in standard.parameters())

        # 深度可分离卷积
        ds_conv = DepthwiseSeparableConv(64, 128, 3, 1, 1)
        ds_params = sum(p.numel() for p in ds_conv.parameters())

        # 深度可分离卷积参数量应该显著减少
        reduction_ratio = ds_params / standard_params
        self.assertLess(reduction_ratio, 0.5)  # 至少减少50%

    def test_ghost_module_efficiency(self):
        """测试Ghost模块效率"""
        from models.edge_optimization import GhostModule

        ghost = GhostModule(64, 128, ratio=2)

        # 计算参数量
        primary_params = sum(p.numel() for p in ghost.primary_conv.parameters())
        cheap_params = sum(p.numel() for p in ghost.cheap_operation.parameters())
        total_params = primary_params + cheap_params

        # 对比标准卷积
        standard = nn.Conv2d(64, 128, 3, 1, 1)
        standard_params = sum(p.numel() for p in standard.parameters())

        # Ghost模块应该减少参数量
        self.assertLess(total_params, standard_params)


class TestEdgeOptimizer(unittest.TestCase):
    """测试边缘优化器"""

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        from models.edge_optimization import EdgeOptimizer

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
        )
        optimizer = EdgeOptimizer(model)

        self.assertIsNotNone(optimizer.model)
        self.assertIsInstance(optimizer.optimization_history, dict)

    def test_benchmark_function(self):
        """测试基准测试功能"""
        from models.edge_optimization import EdgeOptimizer

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        optimizer = EdgeOptimizer(model)

        results = optimizer.benchmark(input_shape=(1, 3, 64, 64), num_runs=3)

        self.assertIn('avg_inference_time_ms', results)
        self.assertIn('fps', results)
        self.assertIn('model_size_mb', results)
        self.assertIn('num_parameters', results)

        self.assertGreater(results['fps'], 0)


class TestIntegration(unittest.TestCase):
    """测试集成"""

    def test_all_edge_configs_present(self):
        """测试所有边缘配置存在"""
        from Config.config import Config

        required_configs = [
            'EDGE_OPTIMIZATION',
            'EDGE_TARGET_PLATFORMS',
            'EDGE_TRAINING_CONFIG',
            'EDGE_COMPRESSION_TARGETS',
            'ONNX_EXPORT_CONFIG',
            'TENSORRT_CONFIG',
        ]

        for config_name in required_configs:
            self.assertTrue(
                hasattr(Config, config_name),
                f"缺少配置: {config_name}"
            )

    def test_edge_target_platforms(self):
        """测试目标平台列表"""
        from Config.config import Config

        platforms = Config.EDGE_TARGET_PLATFORMS
        self.assertIsInstance(platforms, list)
        self.assertGreater(len(platforms), 0)

        # 验证包含常见平台
        expected_platforms = ['cpu', 'gpu']
        for platform in expected_platforms:
            self.assertIn(platform, platforms)


if __name__ == '__main__':
    unittest.main()
