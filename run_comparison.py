"""
模型对比与可视化脚本
用于对比Baseline和改进模型，生成可视化图表
"""
import sys
from pathlib import Path
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO

from Config.config import Config
from evaluation.evaluator import ModelEvaluator
from evaluation.visualizer import ResultVisualizer


def load_models():
    """加载已训练的模型"""
    print("\n" + "=" * 60)
    print("📦 加载模型")
    print("=" * 60)

    models = {}

    # Baseline模型
    baseline_path = Config.OUTPUT_ROOT / "baseline" / "baseline" / "weights" / "best.pt"
    if baseline_path.exists():
        models["Baseline_YOLO11s"] = YOLO(str(baseline_path))
        print(f"   ✅ Baseline: {baseline_path}")
    else:
        print(f"   ⚠️  Baseline模型未找到: {baseline_path}")

    # 改进模型（第三阶段）
    improved_path = Config.OUTPUT_ROOT / "improved" / "stage3_refinement" / "weights" / "best.pt"
    if improved_path.exists():
        models["Improved_MarineYOLO"] = YOLO(str(improved_path))
        print(f"   ✅ Improved: {improved_path}")
    else:
        print(f"   ⚠️  改进模型未找到: {improved_path}")

    return models


def evaluate_and_compare(models: dict):
    """评估并对比模型"""
    print("\n" + "=" * 60)
    print("📊 评估与对比")
    print("=" * 60)

    data_yaml = "./Config/ship_detection.yaml"
    results = {}

    # 评估每个模型
    for name, model in models.items():
        print(f"\n   评估 {name}...")
        evaluator = ModelEvaluator(model, name)
        metrics = evaluator.evaluate(data_yaml, save_results=True)
        results[name] = metrics

    # 如果两个模型都存在，进行对比
    if len(models) == 2:
        model_names = list(models.keys())
        baseline_name = [n for n in model_names if "Baseline" in n][0]
        improved_name = [n for n in model_names if "Improved" in n][0]

        baseline_model = models[baseline_name]
        improved_model = models[improved_name]

        # 对比
        improved_evaluator = ModelEvaluator(improved_model, improved_name)
        comparison = improved_evaluator.compare_with_baseline(baseline_model, data_yaml)

        # 保存对比结果
        output_dir = Config.get_output_dir("results")
        with open(output_dir / "comparison_report.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        return results, comparison

    return results, None


def generate_visualizations(results: dict, comparison: dict = None):
    """生成可视化图表"""
    print("\n" + "=" * 60)
    print("📈 生成可视化图表")
    print("=" * 60)

    visualizer = ResultVisualizer()
    visualizer.generate_report(results, comparison)

    print("\n✅ 可视化完成")


def print_comparison_table(comparison: dict):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("📊 模型性能对比")
    print("=" * 80)

    print(f"\n{'指标':<20} {'Baseline':<12} {'Improved':<12} {'绝对提升':<12} {'相对提升':<12}")
    print("-" * 80)

    for metric, values in comparison.items():
        baseline = values['baseline']
        improved = values['improved']
        abs_imp = values['absolute_improvement']
        rel_imp = values['relative_improvement_percent']

        print(f"{metric:<20} {baseline:<12.4f} {improved:<12.4f} "
              f"{abs_imp:+<12.4f} {rel_imp:+<12.2f}%")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚢 海洋舰船检测 - 模型对比与可视化")
    print("=" * 80)

    # 加载模型
    models = load_models()

    if not models:
        print("\n❌ 未找到任何模型，请先运行训练")
        return

    # 评估和对比
    results, comparison = evaluate_and_compare(models)

    # 生成可视化
    generate_visualizations(results, comparison)

    # 打印对比表格
    if comparison:
        print_comparison_table(comparison)

    # 输出路径
    print("\n" + "=" * 80)
    print("✅ 对比完成！")
    print("=" * 80)
    print(f"\n结果目录: {Config.OUTPUT_ROOT}")
    print(f"   评估结果: {Config.OUTPUT_ROOT}/results/")
    print(f"   可视化图表: {Config.OUTPUT_ROOT}/visualization/")


if __name__ == "__main__":
    main()
