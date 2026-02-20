"""
模型评估器
评估模型性能并对比
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

from ultralytics import YOLO

from Config.config import Config


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model: YOLO, model_name: str):
        self.model = model
        self.model_name = model_name
        self.results = {}

    def evaluate(
        self,
        data_path: str,
        save_results: bool = True,
        result_dir: str = "results"
    ) -> Dict:
        """
        评估模型

        Args:
            data_path: 数据配置文件路径
            save_results: 是否保存结果
            result_dir: 结果保存目录

        Returns:
            评估指标字典
        """
        print(f"\n   评估 {self.model_name}...")

        # 运行验证
        results = self.model.val(
            data=data_path,
            imgsz=Config.TARGET_SIZE,
            batch=Config.BATCH_SIZE,
            device=Config.DEVICE,
            task='obb',
            verbose=False
        )

        # 提取指标
        self.results = {
            "model_name": self.model_name,
            "mAP50": float(results.results_dict.get('metrics/mAP50', 0)),
            "mAP50_95": float(results.results_dict.get('metrics/mAP50-95', 0)),
            "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
            "recall": float(results.results_dict.get('metrics/recall(B)', 0)),
            "fitness": float(results.results_dict.get('fitness', 0)),
        }

        print(f"      mAP50: {self.results['mAP50']:.4f}")
        print(f"      mAP50-95: {self.results['mAP50_95']:.4f}")
        print(f"      Precision: {self.results['precision']:.4f}")
        print(f"      Recall: {self.results['recall']:.4f}")

        # 保存结果
        if save_results:
            self._save_results(result_dir)

        return self.results

    def _save_results(self, result_dir: str):
        """保存评估结果"""
        output_dir = Config.get_output_dir(result_dir)
        output_file = output_dir / f"{self.model_name}_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"      结果已保存: {output_file}")

    def compare_with_baseline(self, baseline_model: YOLO, data_path: str) -> Dict:
        """
        与基线模型对比

        Args:
            baseline_model: 基线模型
            data_path: 数据路径

        Returns:
            对比结果
        """
        # 评估基线
        baseline_evaluator = ModelEvaluator(baseline_model, "Baseline")
        baseline_results = baseline_evaluator.evaluate(data_path, save_results=False)

        # 计算改进
        comparison = {}
        for key in ['mAP50', 'mAP50_95', 'precision', 'recall']:
            baseline_val = baseline_results.get(key, 0)
            improved_val = self.results.get(key, 0)

            comparison[key] = {
                'baseline': baseline_val,
                'improved': improved_val,
                'absolute_improvement': improved_val - baseline_val,
                'relative_improvement_percent': (
                    (improved_val - baseline_val) / baseline_val * 100
                    if baseline_val > 0 else 0
                )
            }

        return comparison


def compare_models(
    models: Dict[str, YOLO],
    data_path: str,
    output_dir: str = "results"
) -> Dict:
    """
    对比多个模型

    Args:
        models: 模型字典 {name: model}
        data_path: 数据路径
        output_dir: 输出目录

    Returns:
        对比结果
    """
    print("\n" + "=" * 60)
    print("📊 模型对比")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        evaluator = ModelEvaluator(model, name)
        results[name] = evaluator.evaluate(data_path, save_results=True, result_dir=output_dir)

    # 保存对比结果
    output_path = Config.get_output_dir(output_dir) / "comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 对比结果已保存: {output_path}")

    return results
