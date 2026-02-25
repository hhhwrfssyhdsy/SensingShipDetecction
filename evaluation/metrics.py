"""
评估指标计算模块
"""
import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个框的IoU

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    计算Average Precision

    Args:
        recalls: 召回率数组
        precisions: 精确率数组

    Returns:
        AP值
    """
    # 使用11点插值法
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    计算mAP

    Args:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
        iou_threshold: IoU阈值

    Returns:
        包含mAP等指标的字典
    """
    # 这里简化实现，实际使用Ultralytics的评估功能
    return {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }


def format_metrics(metrics: Dict) -> str:
    """格式化指标输出"""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
