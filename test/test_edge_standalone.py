"""
边缘计算优化模块独立测试 - 不依赖外部库
"""
import unittest
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEdgeConfig(unittest.TestCase):
    """测试边缘计算配置"""

    def test_edge_optimization_config_exists(self):
        """测试边缘优化配置存在"""
        from Config.config import Config
        self.assertTrue(hasattr(Config, 'EDGE_OPTIMIZATION'))
        self.assertIn('enabled', Config.EDGE_OPTIMIZATION)

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

    def test_quantization_bits(self):
        """测试量化位数"""
        from Config.config import Config

        bits = Config.EDGE_OPTIMIZATION["quantization_bits"]
        self.assertIn(bits, [8, 16, 32])

    def test_edge_training_config(self):
        """测试边缘训练配置"""
        from Config.config import Config

        edge_training = Config.EDGE_TRAINING_CONFIG
        self.assertIn("epochs", edge_training)
        self.assertIn("batch_size", edge_training)
        self.assertIn("learning_rate", edge_training)
        self.assertIn("weight_decay", edge_training)
        self.assertIn("label_smoothing", edge_training)
        self.assertIn("dropout", edge_training)

        # 验证数值
        self.assertGreater(edge_training["epochs"], 0)
        self.assertGreater(edge_training["batch_size"], 0)
        self.assertGreater(edge_training["learning_rate"], 0)

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

        # 验证FPS和延迟关系
        expected_latency = 1000 / targets["min_fps"]  # ms
        self.assertAlmostEqual(targets["target_latency_ms"], expected_latency, delta=5)

    def test_onnx_export_config(self):
        """测试ONNX导出配置"""
        from Config.config import Config

        onnx_config = Config.ONNX_EXPORT_CONFIG
        self.assertIn("opset_version", onnx_config)
        self.assertIn("dynamic_axes", onnx_config)
        self.assertIn("simplify", onnx_config)

        self.assertIsInstance(onnx_config["opset_version"], int)
        self.assertIsInstance(onnx_config["dynamic_axes"], bool)
        self.assertIsInstance(onnx_config["simplify"], bool)

    def test_tensorrt_config(self):
        """测试TensorRT配置"""
        from Config.config import Config

        trt_config = Config.TENSORRT_CONFIG
        self.assertIn("fp16_mode", trt_config)
        self.assertIn("max_batch_size", trt_config)
        self.assertIn("max_workspace_size", trt_config)

        self.assertIsInstance(trt_config["fp16_mode"], bool)
        self.assertIsInstance(trt_config["max_batch_size"], int)
        self.assertIsInstance(trt_config["max_workspace_size"], int)

    def test_edge_target_platforms(self):
        """测试目标平台"""
        from Config.config import Config

        platforms = Config.EDGE_TARGET_PLATFORMS
        self.assertIsInstance(platforms, list)
        self.assertGreater(len(platforms), 0)

        # 验证包含必要平台
        self.assertIn("cpu", platforms)
        self.assertIn("gpu", platforms)


class TestEdgeModules(unittest.TestCase):
    """测试边缘计算模块（直接导入，不通过__init__）"""

    def test_depthwise_separable_conv(self):
        """测试深度可分离卷积"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            DepthwiseSeparableConv = module.DepthwiseSeparableConv

            # 测试初始化
            conv = DepthwiseSeparableConv(64, 128, 3, 1, 1)
            self.assertIsNotNone(conv)

            # 测试前向传播
            x = torch.randn(2, 64, 32, 32)
            out = conv(x)
            self.assertEqual(out.shape, (2, 128, 32, 32))

        except Exception as e:
            self.fail(f"测试失败: {e}")

    def test_ghost_module(self):
        """测试Ghost模块"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            GhostModule = module.GhostModule

            # 测试初始化
            ghost = GhostModule(64, 128, ratio=2)
            self.assertIsNotNone(ghost)

            # 测试前向传播
            x = torch.randn(2, 64, 32, 32)
            out = ghost(x)
            self.assertEqual(out.shape, (2, 128, 32, 32))

        except Exception as e:
            self.fail(f"测试失败: {e}")

    def test_lightweight_bottleneck(self):
        """测试轻量化Bottleneck"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            LightweightBottleneck = module.LightweightBottleneck

            # 测试带shortcut
            bottleneck = LightweightBottleneck(64, 64, shortcut=True)
            x = torch.randn(2, 64, 32, 32)
            out = bottleneck(x)
            self.assertEqual(out.shape, x.shape)

            # 测试不带shortcut
            bottleneck2 = LightweightBottleneck(64, 128, shortcut=False)
            out2 = bottleneck2(x)
            self.assertEqual(out2.shape, (2, 128, 32, 32))

        except Exception as e:
            self.fail(f"测试失败: {e}")


class TestParameterReduction(unittest.TestCase):
    """测试参数量减少"""

    def test_depthwise_parameter_count(self):
        """测试深度可分离卷积参数量"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DepthwiseSeparableConv = module.DepthwiseSeparableConv

        # 标准卷积参数量: 64 * 128 * 3 * 3 + 128 = 73,856
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
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        GhostModule = module.GhostModule

        ghost = GhostModule(64, 128, ratio=2)

        # 计算参数量
        total_params = sum(p.numel() for p in ghost.parameters())

        # 对比标准卷积
        standard = nn.Conv2d(64, 128, 3, 1, 1)
        standard_params = sum(p.numel() for p in standard.parameters())

        # Ghost模块应该减少参数量
        self.assertLess(total_params, standard_params)


class TestEdgeOptimizer(unittest.TestCase):
    """测试边缘优化器"""

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EdgeOptimizer = module.EdgeOptimizer

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
        )
        optimizer = EdgeOptimizer(model)

        self.assertIsNotNone(optimizer.model)
        self.assertIsInstance(optimizer.optimization_history, dict)

    def test_benchmark_function(self):
        """测试基准测试功能"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "edge_optimization",
            Path(__file__).parent.parent / "models" / "edge_optimization.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EdgeOptimizer = module.EdgeOptimizer

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
        self.assertGreater(results['num_parameters'], 0)


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

    def test_edge_enabled_type(self):
        """测试边缘优化开关类型"""
        from Config.config import Config

        self.assertIsInstance(Config.EDGE_OPTIMIZATION['enabled'], bool)

    def test_trainer_edge_method_exists(self):
        """测试训练器方法存在（通过代码检查）"""
        trainer_file = Path(__file__).parent.parent / "trainer.py"
        content = trainer_file.read_text(encoding='utf-8')

        self.assertIn("train_edge_optimized", content)
        self.assertIn("EdgeOptimizer", content)


if __name__ == '__main__':
    unittest.main()
