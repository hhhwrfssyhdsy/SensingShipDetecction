"""
结果可视化模块
生成对比图表
"""
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from Config.config import Config


class ResultVisualizer:
    """结果可视化器"""

    def __init__(self, output_dir: str = "visualization"):
        self.output_dir = Config.get_output_dir(output_dir)
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """设置中文字体"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_comparison_bar(self, results: Dict[str, Dict]):
        """
        绘制对比柱状图

        Args:
            results: 结果字典 {model_name: metrics}
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('YOLO11 海洋舰船检测 - 模型性能对比', fontsize=16)

        metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
        titles = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            models = list(results.keys())
            values = [results[m].get(metric, 0) for m in models]

            colors = ['#3498db', '#e74c3c'][:len(models)]
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)

            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "comparison_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   保存对比图: {output_path}")
        plt.close()

    def plot_radar(self, results: Dict[str, Dict]):
        """
        绘制雷达图

        Args:
            results: 结果字典
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
        titles = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#3498db', '#e74c3c']

        for idx, (model_name, metrics_dict) in enumerate(results.items()):
            values = [metrics_dict.get(m, 0) for m in metrics]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(titles)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        output_path = self.output_dir / "comparison_radar.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   保存雷达图: {output_path}")
        plt.close()

    def plot_improvement(self, comparison: Dict):
        """
        绘制改进幅度图

        Args:
            comparison: 对比结果
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = list(comparison.keys())
        improvements = [comparison[m]['relative_improvement_percent'] for m in metrics]

        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.barh(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:+.2f}%',
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=10)

        ax.set_xlabel('改进幅度 (%)', fontsize=12)
        ax.set_title('改进模型相对提升幅度', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "improvement_percentage.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   保存改进图: {output_path}")
        plt.close()

    def generate_report(self, results: Dict[str, Dict], comparison: Dict = None):
        """
        生成综合报告

        Args:
            results: 结果字典
            comparison: 对比结果
        """
        # 生成所有图表
        self.plot_comparison_bar(results)
        self.plot_radar(results)

        if comparison:
            self.plot_improvement(comparison)

        print(f"\n✅ 所有可视化图表已保存到: {self.output_dir}")


def plot_training_curves(results_dir: Path, output_path: Path = None):
    """
    绘制训练曲线

    Args:
        results_dir: 训练结果目录
        output_path: 输出路径
    """
    # 简化实现，实际从results.csv读取
    print(f"   训练曲线图功能待实现")
    pass
