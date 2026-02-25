"""
Trainer模块单元测试
"""
import unittest
import sys
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer import ShipDetectionTrainer
from Config.config import Config


class TestShipDetectionTrainer(unittest.TestCase):
    """测试训练器"""

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = ShipDetectionTrainer()
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.data_path, "./Config/ship_detection.yaml")

    def test_trainer_with_custom_data_path(self):
        """测试自定义数据路径"""
        custom_path = "./custom/data.yaml"
        trainer = ShipDetectionTrainer(data_path=custom_path)
        self.assertEqual(trainer.data_path, custom_path)

    def test_output_root(self):
        """测试输出根目录"""
        trainer = ShipDetectionTrainer()
        self.assertEqual(trainer.output_root, Config.OUTPUT_ROOT)

    def test_training_history_initial(self):
        """测试初始训练历史"""
        trainer = ShipDetectionTrainer()
        self.assertIsInstance(trainer.training_history, dict)
        self.assertEqual(len(trainer.training_history), 0)

    def test_stage_config_count(self):
        """测试阶段配置数量"""
        stages = Config.PROGRESSIVE_STAGES
        self.assertEqual(len(stages), 3)

    def test_stage_epochs_sum(self):
        """测试阶段轮数总和"""
        stages = Config.PROGRESSIVE_STAGES
        total_epochs = sum(stage['epochs'] for stage in stages)
        self.assertEqual(total_epochs, Config.BASELINE_EPOCHS)

    def test_stage_names(self):
        """测试阶段名称"""
        stages = Config.PROGRESSIVE_STAGES
        expected_names = ['stage1_foundation', 'stage2_enhancement', 'stage3_refinement']
        for i, stage in enumerate(stages):
            self.assertEqual(stage['name'], expected_names[i])


class TestTrainingConfiguration(unittest.TestCase):
    """测试训练配置"""

    def test_stage1_config(self):
        """测试Stage 1配置"""
        stage = Config.PROGRESSIVE_STAGES[0]
        self.assertEqual(stage['epochs'], 60)
        self.assertEqual(stage['attention'], 'se')
        self.assertEqual(stage['mosaic'], 0.2)
        self.assertEqual(stage['mixup'], 0.02)
        self.assertEqual(stage['box'], 5.0)

    def test_stage2_config(self):
        """测试Stage 2配置"""
        stage = Config.PROGRESSIVE_STAGES[1]
        self.assertEqual(stage['epochs'], 40)
        self.assertEqual(stage['attention'], 'cbam')
        self.assertEqual(stage['mosaic'], 0.5)
        self.assertEqual(stage['mixup'], 0.1)
        self.assertEqual(stage['box'], 6.5)

    def test_stage3_config(self):
        """测试Stage 3配置"""
        stage = Config.PROGRESSIVE_STAGES[2]
        self.assertEqual(stage['epochs'], 30)
        self.assertEqual(stage['attention'], 'marine')
        self.assertEqual(stage['mosaic'], 0.8)
        self.assertEqual(stage['mixup'], 0.2)
        self.assertEqual(stage['box'], 8.0)
        self.assertEqual(stage['copy_paste'], 0.15)

    def test_learning_rate_progression(self):
        """测试学习率递减"""
        stages = Config.PROGRESSIVE_STAGES
        lr_values = [stage['lr0'] for stage in stages]
        # 学习率应该递减
        self.assertGreater(lr_values[0], lr_values[1])
        self.assertGreater(lr_values[1], lr_values[2])


if __name__ == '__main__':
    unittest.main()
