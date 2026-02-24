"""
主程序 - 边缘设备优化版
自动执行：数据准备 -> Baseline训练 -> 改进训练 -> 边缘优化 -> 对比评估 -> 可视化
针对嵌入式边缘设备进行全流程优化
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Config.config import Config
from Data.prepare_dataset import validate_dataset
from Data.dataset_analyzer import analyze_dataset
from trainer import ShipDetectionTrainer
from evaluation.evaluator import ModelEvaluator
from evaluation.visualizer import ResultVisualizer
from models.edge_optimization import EdgeOptimizer
from ultralytics import YOLO


def check_environment():
    """检查环境配置"""
    print("\n" + "=" * 60)
    print("🔍 环境检查")
    print("=" * 60)

    # 检查路径
    if not Config.validate_paths():
        print("\n⚠️  请配置正确的数据集路径")
        print("   可以通过环境变量或修改 Config/config.py 设置")
        return False

    # 检查Ultralytics
    try:
        import ultralytics
        print(f"   Ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("   ❌ 未安装 ultralytics")
        print("   请运行: pip install ultralytics")
        return False

    # 检查matplotlib
    try:
        import matplotlib
        print(f"   Matplotlib 版本: {matplotlib.__version__}")
    except ImportError:
        print("   ⚠️  未安装 matplotlib，可视化功能将不可用")
        print("   请运行: pip install matplotlib")

    # 检查边缘优化配置
    print("\n   边缘优化配置:")
    edge_config = Config.EDGE_OPTIMIZATION
    print(f"      启用状态: {edge_config['enabled']}")
    print(f"      轻量化模块: {edge_config['use_lightweight_blocks']}")
    print(f"      模型剪枝: {edge_config['use_pruning']} (比例: {edge_config['pruning_ratio']})")
    print(f"      量化: {edge_config['use_quantization']} ({edge_config['quantization_bits']}bit)")

    print("\n✅ 环境检查通过")
    return True


def run_full_pipeline():

    # 1. 检查环境
    if not check_environment():
        return

    # 2. 验证数据集
    dataset_result = validate_dataset()
    if not dataset_result["valid"]:
        print("\n❌ 数据集验证失败，请检查数据集结构")
        return

    # 3. 分析数据集
    analyze_dataset()

    # 4. 创建训练器
    data_yaml = "./Config/ship_detection.yaml"
    trainer = ShipDetectionTrainer(data_yaml)

    # 检查训练状态
    baseline_trained, baseline_path = trainer.check_baseline_trained()
    improved_trained, improved_path = trainer.check_improved_trained()

    if baseline_trained and improved_trained:
        print("\n" + "=" * 60)
        print("📋 训练状态检查")
        print("=" * 60)
        print(f"   ✅ Baseline 模型已训练: {baseline_path}")
        print(f"   ✅ 改进模型已训练完成: {improved_path}")
        print(f"\n   检测到已训练的模型，将跳过训练阶段")
        print("   如需重新训练，请删除输出目录中的模型文件")
        print("=" * 60)

    # 5. 训练Baseline
    print("\nStep 1: 训练 Baseline 模型")
    print("=" * 60)
    baseline_model, _ = trainer.train_baseline(skip_if_trained=True)

    # 6. 训练改进模型（三阶段渐进训练）
    print("\nStep 2: 训练改进模型 (三阶段渐进训练)")
    improved_model, _ = trainer.train_progressive(skip_if_trained=True)

    # 7. 对比评估
    print("Step 3: 模型对比评估")

    # 评估Baseline
    baseline_evaluator = ModelEvaluator(baseline_model, "Baseline_YOLO11s")
    baseline_metrics = baseline_evaluator.evaluate(data_yaml)

    # 评估改进模型
    improved_evaluator = ModelEvaluator(improved_model, "Improved_MarineYOLO")
    improved_metrics = improved_evaluator.evaluate(data_yaml)

    # 对比
    comparison = improved_evaluator.compare_with_baseline(baseline_model, data_yaml)

    # 8. 导出边缘部署模型
    print("\n" + "=" * 60)
    print("Step 4: 导出边缘部署模型")

    # 获取模型路径
    baseline_model_path = str(Config.OUTPUT_ROOT / "baseline" / "baseline" / "weights" / "best.pt")
    improved_model_path = str(Config.OUTPUT_ROOT / "improved" / "stage3_refinement" / "weights" / "best.pt")

    # 导出模型
    export_results = trainer.export_for_edge_deployment(
        baseline_model_path=baseline_model_path,
        improved_model_path=improved_model_path
    )

    # 9. 可视化
    print("Step 5: 生成可视化图表")

    visualizer = ResultVisualizer()

    results = {
        "Baseline_YOLO11s": baseline_metrics,
        "Improved_MarineYOLO": improved_metrics
    }

    visualizer.generate_report(results, comparison)

    # 10. 打印最终对比结果
    print("最终对比结果")

    print("\n精度指标对比:")
    print(f"{'指标':<20} {'Baseline':<15} {'Improved':<15} {'提升':<15}")
    print("-" * 65)

    for metric in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        baseline_val = baseline_metrics.get(metric, 0)
        improved_val = improved_metrics.get(metric, 0)
        abs_imp = improved_val - baseline_val
        rel_imp = (abs_imp / baseline_val * 100) if baseline_val > 0 else 0

        print(f"{metric:<20} {baseline_val:<15.4f} {improved_val:<15.4f} "
              f"{abs_imp:+.4f} ({rel_imp:+.2f}%)")

    # 11. 输出路径信息
    print(f"\n输出目录: {Config.OUTPUT_ROOT}")
    print("\n模型文件:")
    print(f"   Baseline:    {Config.OUTPUT_ROOT}/baseline/baseline/weights/best.pt")
    print(f"   Improved:    {Config.OUTPUT_ROOT}/improved/stage3_refinement/weights/best.pt")
    print("\n边缘部署文件:")
    print(f"   Baseline ONNX:    {Config.OUTPUT_ROOT}/edge_deployment/baseline/baseline_yolo11s.onnx")
    print(f"   Improved ONNX:    {Config.OUTPUT_ROOT}/edge_deployment/improved/improved_yolo11s.onnx")
    print("\n结果文件:")
    print(f"   评估结果:    {Config.OUTPUT_ROOT}/results/")
    print(f"   可视化图表:  {Config.OUTPUT_ROOT}/visualization/")
    print(f"   训练历史:    {Config.OUTPUT_ROOT}/training_history.json")

def main():
    """主函数"""
    try:
        run_full_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
