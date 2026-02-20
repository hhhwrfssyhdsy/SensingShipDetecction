"""
Config模块单元测试
"""
import unittest
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from Config.config import Config


class TestConfig(unittest.TestCase):
    """测试配置类"""

    def test_config_exists(self):
        """测试配置类是否存在"""
        self.assertIsNotNone(Config)

    def test_basic_config_values(self):
        """测试基本配置值"""
        # 测试训练参数
        self.assertIsInstance(Config.TARGET_SIZE, int)
        self.assertGreater(Config.TARGET_SIZE, 0)
        self.assertIsInstance(Config.BATCH_SIZE, int)
        self.assertGreater(Config.BATCH_SIZE, 0)

    def test_baseline_epochs(self):
        """测试Baseline训练轮数"""
        self.assertEqual(Config.BASELINE_EPOCHS, 130)

    def test_progressive_stages(self):
        """测试渐进训练阶段配置"""
        stages = Config.PROGRESSIVE_STAGES
        self.assertIsInstance(stages, list)
        self.assertEqual(len(stages), 3)

        # 检查每个阶段的配置
        total_epochs = 0
        for i, stage in enumerate(stages):
            self.assertIn('name', stage)
            self.assertIn('epochs', stage)
            self.assertIn('attention', stage)
            self.assertIn('description', stage)
            total_epochs += stage['epochs']

        # 验证总轮数
        self.assertEqual(total_epochs, 130)

    def test_stage_attention_types(self):
        """测试各阶段的注意力类型"""
        stages = Config.PROGRESSIVE_STAGES
        expected_attentions = ['se', 'cbam', 'marine']

        for i, stage in enumerate(stages):
            self.assertEqual(stage['attention'], expected_attentions[i])

    def test_stage_data_augmentation(self):
        """测试各阶段的数据增强配置"""
        stages = Config.PROGRESSIVE_STAGES

        # Stage 1: 保守增强
        self.assertEqual(stages[0]['mosaic'], 0.2)
        self.assertEqual(stages[0]['mixup'], 0.02)

        # Stage 2: 中等增强
        self.assertEqual(stages[1]['mosaic'], 0.5)
        self.assertEqual(stages[1]['mixup'], 0.1)

        # Stage 3: 强增强
        self.assertEqual(stages[2]['mosaic'], 0.8)
        self.assertEqual(stages[2]['mixup'], 0.2)
        self.assertEqual(stages[2]['copy_paste'], 0.15)

    def test_unified_classes(self):
        """测试统一类别配置"""
        self.assertIn('ship', Config.UNIFIED_CLASSES)
        self.assertEqual(Config.UNIFIED_CLASSES['ship'], 0)

    def test_output_root(self):
        """测试输出根目录"""
        self.assertIsInstance(Config.OUTPUT_ROOT, Path)

    def test_get_output_dir(self):
        """测试获取输出目录方法"""
        output_dir = Config.get_output_dir("test_subdir")
        self.assertIsInstance(output_dir, Path)
        self.assertTrue(output_dir.exists())


if __name__ == '__main__':
    unittest.main()
