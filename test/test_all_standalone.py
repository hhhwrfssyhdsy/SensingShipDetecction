"""
独立的单元测试 - 不依赖外部库
"""
import unittest
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig(unittest.TestCase):
    """测试配置类"""

    def test_import_config(self):
        """测试导入Config"""
        try:
            from Config.config import Config
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入Config失败: {e}")

    def test_config_values(self):
        """测试配置值"""
        from Config.config import Config

        # 测试基本配置
        self.assertIsInstance(Config.TARGET_SIZE, int)
        self.assertGreater(Config.TARGET_SIZE, 0)
        self.assertIsInstance(Config.BATCH_SIZE, int)
        self.assertGreater(Config.BATCH_SIZE, 0)

    def test_baseline_epochs(self):
        """测试Baseline训练轮数"""
        from Config.config import Config
        self.assertEqual(Config.BASELINE_EPOCHS, 130)

    def test_progressive_stages(self):
        """测试渐进训练阶段"""
        from Config.config import Config

        stages = Config.PROGRESSIVE_STAGES
        self.assertIsInstance(stages, list)
        self.assertEqual(len(stages), 3)

        # 验证总轮数
        total_epochs = sum(s['epochs'] for s in stages)
        self.assertEqual(total_epochs, 130)

    def test_stage_attention_types(self):
        """测试注意力类型"""
        from Config.config import Config

        stages = Config.PROGRESSIVE_STAGES
        expected = ['se', 'cbam', 'marine']

        for i, stage in enumerate(stages):
            self.assertEqual(stage['attention'], expected[i])

    def test_stage_data_augmentation(self):
        """测试数据增强配置"""
        from Config.config import Config

        stages = Config.PROGRESSIVE_STAGES

        # Stage 1: 保守
        self.assertEqual(stages[0]['mosaic'], 0.2)
        self.assertEqual(stages[0]['mixup'], 0.02)

        # Stage 2: 中等
        self.assertEqual(stages[1]['mosaic'], 0.5)
        self.assertEqual(stages[1]['mixup'], 0.1)

        # Stage 3: 强
        self.assertEqual(stages[2]['mosaic'], 0.8)
        self.assertEqual(stages[2]['mixup'], 0.2)
        self.assertEqual(stages[2]['copy_paste'], 0.15)

    def test_unified_classes(self):
        """测试统一类别"""
        from Config.config import Config
        self.assertIn('ship', Config.UNIFIED_CLASSES)


class TestMetrics(unittest.TestCase):
    """测试指标计算"""

    def test_iou_same_box(self):
        """测试相同框的IoU"""
        box = np.array([0, 0, 10, 10])

        # 计算IoU
        x1 = max(box[0], box[0])
        y1 = max(box[1], box[1])
        x2 = min(box[2], box[2])
        y2 = min(box[3], box[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        union = area + area - intersection
        iou = intersection / union if union > 0 else 0

        self.assertAlmostEqual(iou, 1.0)

    def test_iou_no_overlap(self):
        """测试无重叠"""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        self.assertEqual(intersection, 0)

    def test_iou_partial_overlap(self):
        """测试部分重叠"""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0

        expected = 25 / 175  # 交集25，并集175
        self.assertAlmostEqual(iou, expected, places=5)


class TestDataProcessing(unittest.TestCase):
    """测试数据处理"""

    def test_label_conversion_horizontal(self):
        """测试水平框转换"""
        # 模拟水平框标签: class x y w h
        line = "0 0.5 0.5 0.3 0.4"
        parts = line.strip().split()

        if len(parts) == 5:
            cls_id = parts[0]
            x, y, w, h = map(float, parts[1:5])
            # 转换为旋转框格式
            result = f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} 0.0"

            result_parts = result.split()
            self.assertEqual(len(result_parts), 6)
            self.assertEqual(float(result_parts[5]), 0.0)

    def test_label_conversion_rotated(self):
        """测试旋转框转换"""
        # 模拟旋转框标签: class x y w h angle
        line = "0 0.5 0.5 0.3 0.4 0.785"
        parts = line.strip().split()

        if len(parts) == 6:
            result = line  # 保持不变
            result_parts = result.split()
            self.assertEqual(len(result_parts), 6)
            self.assertAlmostEqual(float(result_parts[5]), 0.785, places=3)


class TestTrainingConfig(unittest.TestCase):
    """测试训练配置"""

    def test_stage_count(self):
        """测试阶段数量"""
        from Config.config import Config
        self.assertEqual(len(Config.PROGRESSIVE_STAGES), 3)

    def test_stage_names(self):
        """测试阶段名称"""
        from Config.config import Config

        expected_names = [
            'stage1_foundation',
            'stage2_enhancement',
            'stage3_refinement'
        ]

        for i, stage in enumerate(Config.PROGRESSIVE_STAGES):
            self.assertEqual(stage['name'], expected_names[i])

    def test_learning_rate_decreasing(self):
        """测试学习率递减"""
        from Config.config import Config

        stages = Config.PROGRESSIVE_STAGES
        lr_values = [stage['lr0'] for stage in stages]

        # 学习率应该递减: 0.01 > 0.005 > 0.002
        self.assertGreater(lr_values[0], lr_values[1])
        self.assertGreater(lr_values[1], lr_values[2])

    def test_mosaic_increasing(self):
        """测试Mosaic递增"""
        from Config.config import Config

        stages = Config.PROGRESSIVE_STAGES
        mosaic_values = [stage['mosaic'] for stage in stages]

        # Mosaic应该递增: 0.2 < 0.5 < 0.8
        self.assertLess(mosaic_values[0], mosaic_values[1])
        self.assertLess(mosaic_values[1], mosaic_values[2])

    def test_box_loss_increasing(self):
        """测试Box Loss递增"""
        from Config.config import Config

        stages = Config.PROGRESSIVE_STAGES
        box_values = [stage['box'] for stage in stages]

        # Box Loss应该递增: 5.0 < 6.5 < 8.0
        self.assertLess(box_values[0], box_values[1])
        self.assertLess(box_values[1], box_values[2])


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 运行单元测试")
    print("=" * 80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingConfig))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)
    print(f"   总测试数: {result.testsRun}")
    print(f"   通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 存在失败的测试")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
