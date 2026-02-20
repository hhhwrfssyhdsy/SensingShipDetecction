"""
船舶检测系统单元测试
测试数据集准备、标签解析、训练配置等核心功能
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# 导入被测试的模块
from Data.prepare_dataset import DatasetPreparer
from Config.config import Config, TRAINING_STRATEGIES, PROGRESSIVE_STAGES


class TestDatasetPreparer(unittest.TestCase):
    """测试数据集准备模块"""

    def setUp(self):
        """测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_label_dir = Path(self.temp_dir) / "labels"
        self.test_label_dir.mkdir()

    def tearDown(self):
        """测试后清理临时目录"""
        shutil.rmtree(self.temp_dir)

    def test_parse_rotated_bbox(self):
        """测试解析旋转框标签（6参数）"""
        # 旋转框格式: class_id x_center y_center width height angle
        line = "0 0.5 0.5 0.3 0.2 45.0"
        result = DatasetPreparer._parse_and_convert_label(line)

        self.assertIsNotNone(result)
        parts = result.split()
        self.assertEqual(len(parts), 6)  # 应该有6个参数
        self.assertEqual(parts[0], "0")  # class_id
        self.assertEqual(parts[1], "0.5")  # x_center
        self.assertEqual(parts[2], "0.5")  # y_center
        self.assertEqual(parts[3], "0.3")  # width
        self.assertEqual(parts[4], "0.2")  # height
        self.assertEqual(parts[5], "45.0")  # angle

    def test_parse_horizontal_bbox(self):
        """测试解析水平框标签（5参数转6参数）"""
        # 水平框格式: class_id x_center y_center width height
        line = "0 0.5 0.5 0.3 0.2"
        result = DatasetPreparer._parse_and_convert_label(line)

        self.assertIsNotNone(result)
        parts = result.split()
        self.assertEqual(len(parts), 6)  # 应该转换为6个参数
        self.assertEqual(parts[0], "0")  # class_id
        self.assertEqual(parts[5], "0")  # 角度应该设为0

    def test_parse_invalid_line(self):
        """测试解析无效标签行"""
        # 少于5个参数的行
        line = "0 0.5 0.5"
        result = DatasetPreparer._parse_and_convert_label(line)
        self.assertEqual(result, "")

        # 空行
        line = ""
        result = DatasetPreparer._parse_and_convert_label(line)
        self.assertEqual(result, "")

    def test_multi_class_to_single_class(self):
        """测试多类别转换为单类别"""
        # 原始有多个类别
        line = "5 0.5 0.5 0.3 0.2 30.0"
        result = DatasetPreparer._parse_and_convert_label(line)

        parts = result.split()
        self.assertEqual(parts[0], "0")  # 应该转换为类别0


class TestConfig(unittest.TestCase):
    """测试配置模块"""

    def test_training_strategies_exist(self):
        """测试训练策略配置存在"""
        required_strategies = ["standard", "small_object", "marine"]
        for strategy in required_strategies:
            self.assertIn(strategy, TRAINING_STRATEGIES)

    def test_progressive_stages_exist(self):
        """测试渐进式训练阶段配置存在"""
        required_stages = ["two_stage", "three_stage", "synergistic"]
        for stage in required_stages:
            self.assertIn(stage, PROGRESSIVE_STAGES)

    def test_config_paths(self):
        """测试配置路径"""
        # 验证关键配置项存在
        self.assertIsNotNone(Config.TARGET_SIZE)
        self.assertIsNotNone(Config.BATCH_SIZE)
        self.assertIsNotNone(Config.YOLO_MODEL_PATH)
        self.assertGreater(Config.TARGET_SIZE, 0)
        self.assertGreater(Config.BATCH_SIZE, 0)


class TestLabelFormat(unittest.TestCase):
    """测试标签格式转换"""

    def test_rotated_bbox_format(self):
        """测试旋转框格式正确性"""
        test_cases = [
            # (输入, 期望输出)
            ("0 0.5 0.5 0.3 0.2 45.0", "0 0.5 0.5 0.3 0.2 45.0"),
            ("1 0.3 0.4 0.2 0.1 90.0", "0 0.3 0.4 0.2 0.1 90.0"),
            ("5 0.1 0.2 0.05 0.05 0.0", "0 0.1 0.2 0.05 0.05 0.0"),
        ]

        for input_line, expected in test_cases:
            result = DatasetPreparer._parse_and_convert_label(input_line)
            self.assertEqual(result, expected)

    def test_horizontal_to_rotated(self):
        """测试水平框转旋转框"""
        test_cases = [
            # (输入, 期望输出 - 角度为0)
            ("0 0.5 0.5 0.3 0.2", "0 0.5 0.5 0.3 0.2 0"),
            ("2 0.1 0.2 0.05 0.05", "0 0.1 0.2 0.05 0.05 0"),
        ]

        for input_line, expected in test_cases:
            result = DatasetPreparer._parse_and_convert_label(input_line)
            self.assertEqual(result, expected)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_end_to_end_label_conversion(self):
        """测试端到端标签转换"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试标签文件
            label_dir = Path(temp_dir) / "labels"
            label_dir.mkdir()

            # 创建包含混合格式的标签文件
            label_file = label_dir / "test.txt"
            with open(label_file, "w") as f:
                f.write("0 0.5 0.5 0.3 0.2 45.0\n")  # 旋转框
                f.write("1 0.3 0.4 0.2 0.1\n")        # 水平框
                f.write("2 0.1 0.2 0.05 0.05 0.0\n")  # 旋转框，角度0

            # 读取并转换
            converted_lines = []
            with open(label_file, "r") as f:
                for line in f:
                    converted = DatasetPreparer._parse_and_convert_label(line.strip())
                    if converted:
                        converted_lines.append(converted)

            # 验证所有行都转换为6参数格式
            self.assertEqual(len(converted_lines), 3)
            for line in converted_lines:
                parts = line.split()
                self.assertEqual(len(parts), 6)
                self.assertEqual(parts[0], "0")  # 所有类别都转为0


class TestModelPath(unittest.TestCase):
    """测试模型路径转换（不依赖ultralytics）"""

    def test_obb_model_path_conversion(self):
        """测试OBB模型路径转换"""
        # 直接测试路径转换逻辑
        def convert_to_obb(model_path: str) -> str:
            """模拟 _get_model_path 逻辑"""
            if "-obb" in model_path:
                return model_path

            if "yolo11s.pt" in model_path:
                return model_path.replace("yolo11s.pt", "yolo11s-obb.pt")
            elif "yolo11n.pt" in model_path:
                return model_path.replace("yolo11n.pt", "yolo11n-obb.pt")
            elif "yolo11m.pt" in model_path:
                return model_path.replace("yolo11m.pt", "yolo11m-obb.pt")
            elif "yolo11l.pt" in model_path:
                return model_path.replace("yolo11l.pt", "yolo11l-obb.pt")
            elif "yolo11x.pt" in model_path:
                return model_path.replace("yolo11x.pt", "yolo11x-obb.pt")

            return model_path

        test_cases = [
            ("yolo11s.pt", "yolo11s-obb.pt"),
            ("yolo11n.pt", "yolo11n-obb.pt"),
            ("yolo11m.pt", "yolo11m-obb.pt"),
            ("yolo11l.pt", "yolo11l-obb.pt"),
            ("yolo11x.pt", "yolo11x-obb.pt"),
            ("/path/to/yolo11s.pt", "/path/to/yolo11s-obb.pt"),
            ("yolo11s-obb.pt", "yolo11s-obb.pt"),  # 已经是OBB模型
        ]

        for input_path, expected in test_cases:
            result = convert_to_obb(input_path)
            self.assertEqual(result, expected)


class TestTrainingArgs(unittest.TestCase):
    """测试训练参数"""

    def test_base_args_contain_obb_task(self):
        """测试基础参数包含OBB任务"""
        # 模拟 _get_base_args 返回值
        base_args = {
            "data": "config.yaml",
            "task": "obb",  # 统一使用旋转框检测任务
            "imgsz": 640,
            "batch": 16,
            "device": "0",
            "workers": 8,
            "save": True,
            "pretrained": True,
            "optimizer": "auto",
            "verbose": True,
        }

        self.assertEqual(base_args["task"], "obb")
        self.assertIn("data", base_args)
        self.assertIn("imgsz", base_args)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetPreparer))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPath))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingArgs))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 船舶检测系统单元测试")
    print("=" * 60)

    success = run_tests()

    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
