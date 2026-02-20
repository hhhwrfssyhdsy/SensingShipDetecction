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
from Data.prepare_dataset import prepare_dataset
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
    """运行完整流程 - 包含边缘优化"""
    print("\n" + "=" * 80)
    print("🚢 海洋舰船检测 - YOLO11 边缘设备优化完整流程")
    print("=" * 80)

    # 1. 检查环境
    if not check_environment():
        return

    # 2. 分析数据集
    analyze_dataset()

    # 3. 准备数据集
    dataset_path = prepare_dataset()

    # 4. 创建训练器
    data_yaml = "./Config/ship_detection.yaml"
    trainer = ShipDetectionTrainer(data_yaml)

    # 5. 训练Baseline
    print("\n" + "=" * 60)
    print("📌 Step 1: 训练 Baseline 模型")
    print("=" * 60)
    baseline_model, _ = trainer.train_baseline()

    # 6. 训练改进模型（三阶段渐进训练）
    print("\n" + "=" * 60)
    print("📌 Step 2: 训练改进模型 (三阶段渐进训练)")
    print("=" * 60)
    improved_model, _ = trainer.train_progressive()

    # 7. 边缘设备优化
    print("\n" + "=" * 60)
    print("📌 Step 3: 边缘设备优化")
    print("=" * 60)
    edge_model, edge_results = trainer.train_edge_optimized()

    # 8. 对比评估
    print("\n" + "=" * 60)
    print("📌 Step 4: 模型对比评估")
    print("=" * 60)

    # 评估Baseline
    baseline_evaluator = ModelEvaluator(baseline_model, "Baseline_YOLO11s")
    baseline_metrics = baseline_evaluator.evaluate(data_yaml)

    # 评估改进模型
    improved_evaluator = ModelEvaluator(improved_model, "Improved_MarineYOLO")
    improved_metrics = improved_evaluator.evaluate(data_yaml)

    # 评估边缘优化模型
    edge_evaluator = ModelEvaluator(edge_model, "EdgeOptimized_YOLO")
    edge_metrics = edge_evaluator.evaluate(data_yaml)

    # 对比
    comparison = improved_evaluator.compare_with_baseline(baseline_model, data_yaml)

    # 9. 边缘设备性能基准测试
    print("\n" + "=" * 60)
    print("📌 Step 5: 边缘设备性能测试")
    print("=" * 60)

    edge_optimizer = EdgeOptimizer(edge_model.model)
    edge_benchmark = edge_optimizer.benchmark(
        input_shape=(1, 3, Config.TARGET_SIZE, Config.TARGET_SIZE),
        num_runs=100
    )

    # 10. 导出边缘部署模型
    print("\n" + "=" * 60)
    print("📌 Step 6: 导出边缘部署模型")
    print("=" * 60)

    edge_output_dir = Config.get_output_dir("edge_deployment")

    # 导出ONNX
    onnx_path = edge_optimizer.export_to_onnx(str(edge_output_dir / "edge_optimized.onnx"))

    # 保存边缘优化结果
    edge_deployment_info = {
        "model_info": {
            "name": "EdgeOptimized_YOLO",
            "description": "针对边缘设备优化的YOLO11模型",
            "optimization_techniques": [
                "轻量化模块 (DepthwiseSeparableConv, GhostModule)",
                "模型剪枝",
                "INT8量化准备",
                "注意力机制 (SE -> CBAM -> MarineContext)"
            ]
        },
        "performance": {
            "accuracy": edge_metrics,
            "benchmark": edge_benchmark,
            "comparison_with_baseline": comparison
        },
        "deployment": {
            "onnx_model": str(onnx_path),
            "pytorch_model": str(edge_output_dir / "edge_optimized.pt"),
            "target_platforms": Config.EDGE_TARGET_PLATFORMS,
            "recommended_platform": "jetson" if edge_benchmark['fps'] > 30 else "raspberry_pi"
        },
        "config": {
            "edge_optimization": Config.EDGE_OPTIMIZATION,
            "compression_targets": Config.EDGE_COMPRESSION_TARGETS
        }
    }

    import json
    with open(edge_output_dir / "deployment_info.json", 'w', encoding='utf-8') as f:
        json.dump(edge_deployment_info, f, indent=2, ensure_ascii=False)

    print(f"\n   ✅ 边缘部署信息已保存: {edge_output_dir}/deployment_info.json")

    # 11. 可视化
    print("\n" + "=" * 60)
    print("📌 Step 7: 生成可视化图表")
    print("=" * 60)

    visualizer = ResultVisualizer()

    results = {
        "Baseline_YOLO11s": baseline_metrics,
        "Improved_MarineYOLO": improved_metrics,
        "EdgeOptimized_YOLO": edge_metrics
    }

    visualizer.generate_report(results, comparison)

    # 12. 打印最终对比结果
    print("\n" + "=" * 80)
    print("📊 最终对比结果")
    print("=" * 80)

    print("\n精度指标对比:")
    print(f"{'指标':<20} {'Baseline':<12} {'Improved':<12} {'Edge':<12} {'提升':<12}")
    print("-" * 80)

    for metric in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        baseline_val = baseline_metrics.get(metric, 0)
        improved_val = improved_metrics.get(metric, 0)
        edge_val = edge_metrics.get(metric, 0)
        abs_imp = edge_val - baseline_val

        print(f"{metric:<20} {baseline_val:<12.4f} {improved_val:<12.4f} "
              f"{edge_val:<12.4f} {abs_imp:+<12.4f}")

    print("\n边缘设备性能:")
    print(f"   平均推理时间: {edge_benchmark['avg_inference_time_ms']:.2f} ms")
    print(f"   FPS: {edge_benchmark['fps']:.2f}")
    print(f"   模型大小: {edge_benchmark['model_size_mb']:.2f} MB")
    print(f"   参数量: {edge_benchmark['num_parameters']:,}")

    # 检查是否满足边缘设备要求
    targets = Config.EDGE_COMPRESSION_TARGETS
    print("\n边缘设备目标检查:")
    print(f"   模型大小目标: {targets['max_model_size_mb']} MB")
    print(f"   实际模型大小: {edge_benchmark['model_size_mb']:.2f} MB")
    print(f"   状态: {'✅ 通过' if edge_benchmark['model_size_mb'] <= targets['max_model_size_mb'] else '⚠️ 超出'}")

    print(f"\n   FPS目标: {targets['min_fps']}")
    print(f"   实际FPS: {edge_benchmark['fps']:.2f}")
    print(f"   状态: {'✅ 通过' if edge_benchmark['fps'] >= targets['min_fps'] else '⚠️ 未达标'}")

    # 13. 输出路径信息
    print("\n" + "=" * 80)
    print("✅ 所有任务完成！")
    print("=" * 80)
    print(f"\n输出目录: {Config.OUTPUT_ROOT}")
    print("\n模型文件:")
    print(f"   Baseline:    {Config.OUTPUT_ROOT}/baseline/baseline/weights/best.pt")
    print(f"   Improved:    {Config.OUTPUT_ROOT}/improved/stage3_refinement/weights/best.pt")
    print(f"   Edge:        {Config.OUTPUT_ROOT}/edge_optimized/base_training/weights/best.pt")
    print("\n边缘部署文件:")
    print(f"   ONNX模型:    {edge_output_dir}/edge_optimized.onnx")
    print(f"   部署信息:    {edge_output_dir}/deployment_info.json")
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
