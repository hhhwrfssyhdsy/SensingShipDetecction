"""
Data模块单元测试
"""
import unittest
import sys
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from Data.prepare_dataset import DatasetPreparer
from Data.dataset_analyzer import DatasetAnalyzer


class TestDatasetPreparer(unittest.TestCase):
    """测试数据集准备器"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.preparer = DatasetPreparer(output_root=Path(self.temp_dir))

    def test_preparer_initialization(self):
        """测试准备器初始化"""
        self.assertIsNotNone(self.preparer)
        self.assertTrue(self.preparer.train_dir.exists())
        self.assertTrue(self.preparer.val_dir.exists())
        self.assertTrue(self.preparer.test_dir.exists())

    def test_directory_structure(self):
        """测试目录结构"""
        # 检查images和labels子目录
        for split_dir in [self.preparer.train_dir, self.preparer.val_dir, self.preparer.test_dir]:
            self.assertTrue((split_dir / "images").exists())
            self.assertTrue((split_dir / "labels").exists())

    def test_process_label_horizontal(self):
        """测试处理水平框标签"""
        # 创建临时标签文件（5参数格式）
        src_label = Path(self.temp_dir) / "test_horizontal.txt"
        dst_label = Path(self.temp_dir) / "test_output.txt"

        with open(src_label, 'w') as f:
            f.write("0 0.5 0.5 0.3 0.4\n")  # class x y w h

        self.preparer._process_label(src_label, dst_label)

        # 验证输出（应该转换为6参数格式）
        with open(dst_label, 'r') as f:
            content = f.read().strip()

        parts = content.split()
        self.assertEqual(len(parts), 6)  # class x y w h angle
        self.assertEqual(parts[0], "0")
        self.assertEqual(float(parts[5]), 0.0)  # angle应该为0

    def test_process_label_rotated(self):
        """测试处理旋转框标签"""
        src_label = Path(self.temp_dir) / "test_rotated.txt"
        dst_label = Path(self.temp_dir) / "test_output.txt"

        with open(src_label, 'w') as f:
            f.write("0 0.5 0.5 0.3 0.4 0.785\n")  # class x y w h angle

        self.preparer._process_label(src_label, dst_label)

        with open(dst_label, 'r') as f:
            content = f.read().strip()

        parts = content.split()
        self.assertEqual(len(parts), 6)
        self.assertAlmostEqual(float(parts[5]), 0.785, places=3)


class TestDatasetAnalyzer(unittest.TestCase):
    """测试数据集分析器"""

    def test_analyzer_initialization(self):
        """测试分析器初始化"""
        analyzer = DatasetAnalyzer()
        self.assertIsNotNone(analyzer)
        self.assertIn('total_images', analyzer.stats)
        self.assertIn('total_labels', analyzer.stats)
        self.assertIn('total_objects', analyzer.stats)

    def test_initial_stats(self):
        """测试初始统计值"""
        analyzer = DatasetAnalyzer()
        self.assertEqual(analyzer.stats['total_images'], 0)
        self.assertEqual(analyzer.stats['total_labels'], 0)
        self.assertEqual(analyzer.stats['total_objects'], 0)


if __name__ == '__main__':
    unittest.main()
