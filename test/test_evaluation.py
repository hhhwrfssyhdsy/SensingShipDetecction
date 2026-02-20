"""
Evaluation模块单元测试
"""
import unittest
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import calculate_iou, calculate_ap, calculate_map
from evaluation.evaluator import ModelEvaluator


class TestMetrics(unittest.TestCase):
    """测试指标计算"""

    def test_calculate_iou_same_box(self):
        """测试相同框的IoU"""
        box = np.array([0, 0, 10, 10])
        iou = calculate_iou(box, box)
        self.assertAlmostEqual(iou, 1.0)

    def test_calculate_iou_no_overlap(self):
        """测试无重叠框的IoU"""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_calculate_iou_partial_overlap(self):
        """测试部分重叠框的IoU"""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])
        iou = calculate_iou(box1, box2)
        # 交集面积 = 5*5 = 25, 并集面积 = 100+100-25 = 175
        expected_iou = 25 / 175
        self.assertAlmostEqual(iou, expected_iou, places=5)

    def test_calculate_ap(self):
        """测试AP计算"""
        recalls = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
        precisions = np.array([1.0, 0.9, 0.85, 0.8, 0.75, 0.7])
        ap = calculate_ap(recalls, precisions)
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_calculate_map(self):
        """测试mAP计算"""
        predictions = [{"class": 0, "bbox": [0, 0, 10, 10], "score": 0.9}]
        ground_truths = [{"class": 0, "bbox": [0, 0, 10, 10]}]

        result = calculate_map(predictions, ground_truths)
        self.assertIn("mAP50", result)
        self.assertIn("mAP50_95", result)


class TestModelEvaluator(unittest.TestCase):
    """测试模型评估器"""

    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        # 由于需要YOLO模型，这里只做简单的初始化测试
        # 实际测试需要模型文件
        pass


if __name__ == '__main__':
    unittest.main()
