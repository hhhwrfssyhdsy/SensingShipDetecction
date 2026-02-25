"""
评估模块
包含模型评估、指标计算和可视化
"""
from .evaluator import ModelEvaluator, compare_models
from .metrics import calculate_iou, calculate_map
from .visualizer import ResultVisualizer, plot_training_curves, plot_stage_comparison

__all__ = [
    "ModelEvaluator",
    "compare_models",
    "calculate_iou",
    "calculate_map",
    "ResultVisualizer",
    "plot_training_curves",
    "plot_stage_comparison",
]
