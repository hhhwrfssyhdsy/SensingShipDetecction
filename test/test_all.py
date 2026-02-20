"""
运行所有单元测试 - 简化版
不依赖外部库
"""
import unittest
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入测试模块
from test.test_config import TestConfig
from test.test_data import TestDatasetPreparer, TestDatasetAnalyzer
from test.test_evaluation import TestMetrics
from test.test_trainer import TestShipDetectionTrainer, TestTrainingConfiguration


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 运行单元测试")
    print("=" * 80)

    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加Config测试
    suite.addTest(TestConfig('test_config_exists'))
    suite.addTest(TestConfig('test_basic_config_values'))
    suite.addTest(TestConfig('test_baseline_epochs'))
    suite.addTest(TestConfig('test_progressive_stages'))
    suite.addTest(TestConfig('test_stage_attention_types'))
    suite.addTest(TestConfig('test_stage_data_augmentation'))
    suite.addTest(TestConfig('test_unified_classes'))
    suite.addTest(TestConfig('test_output_root'))
    suite.addTest(TestConfig('test_get_output_dir'))

    # 添加Data测试
    suite.addTest(TestDatasetPreparer('test_preparer_initialization'))
    suite.addTest(TestDatasetPreparer('test_directory_structure'))
    suite.addTest(TestDatasetPreparer('test_process_label_horizontal'))
    suite.addTest(TestDatasetPreparer('test_process_label_rotated'))
    suite.addTest(TestDatasetAnalyzer('test_analyzer_initialization'))
    suite.addTest(TestDatasetAnalyzer('test_initial_stats'))

    # 添加Evaluation测试
    suite.addTest(TestMetrics('test_calculate_iou_same_box'))
    suite.addTest(TestMetrics('test_calculate_iou_no_overlap'))
    suite.addTest(TestMetrics('test_calculate_iou_partial_overlap'))
    suite.addTest(TestMetrics('test_calculate_ap'))
    suite.addTest(TestMetrics('test_calculate_map'))

    # 添加Trainer测试
    suite.addTest(TestShipDetectionTrainer('test_trainer_initialization'))
    suite.addTest(TestShipDetectionTrainer('test_trainer_with_custom_data_path'))
    suite.addTest(TestShipDetectionTrainer('test_output_root'))
    suite.addTest(TestShipDetectionTrainer('test_training_history_initial'))
    suite.addTest(TestShipDetectionTrainer('test_stage_config_count'))
    suite.addTest(TestShipDetectionTrainer('test_stage_epochs_sum'))
    suite.addTest(TestShipDetectionTrainer('test_stage_names'))
    suite.addTest(TestTrainingConfiguration('test_stage1_config'))
    suite.addTest(TestTrainingConfiguration('test_stage2_config'))
    suite.addTest(TestTrainingConfiguration('test_stage3_config'))
    suite.addTest(TestTrainingConfiguration('test_learning_rate_progression'))

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
        if result.failures:
            print("\n失败的测试:")
            for test, trace in result.failures:
                print(f"   - {test}")
        if result.errors:
            print("\n错误的测试:")
            for test, trace in result.errors:
                print(f"   - {test}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
