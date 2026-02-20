"""
运行所有单元测试
"""
import unittest
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入所有测试模块
from test import (
    test_config,
    test_data,
    test_models,
    test_evaluation,
    test_trainer,
)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 运行单元测试")
    print("=" * 80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试模块
    suite.addTests(loader.loadTestsFromModule(test_config))
    suite.addTests(loader.loadTestsFromModule(test_data))
    suite.addTests(loader.loadTestsFromModule(test_models))
    suite.addTests(loader.loadTestsFromModule(test_evaluation))
    suite.addTests(loader.loadTestsFromModule(test_trainer))

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
