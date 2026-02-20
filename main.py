"""
船舶目标检测训练主程序
支持自动数据集分析、多策略训练和渐进式优化
"""
import argparse
import sys
from pathlib import Path

from Config.config import Config
from Data.dataset_analyzer import analyze_marine_dataset
from trainer import ShipDetectionTrainer
from Data.prepare_dataset import prepare_ship_dataset



def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")

    # 检查路径配置
    valid, errors = Config.validate_paths()
    if not valid:
        print("⚠️  部分路径验证失败:")
        for error in errors:
            print(f"   - {error}")
        print("\n提示: 可以通过环境变量覆盖默认路径")
        return False

    print("✅ 环境检查通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="船舶目标检测训练系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 自动选择策略
  python main.py --strategy standard # 使用标准训练
  python main.py --prepare-only     # 仅准备数据集
  python main.py --analyze-only     # 仅分析数据集

可用策略:
  - auto: 自动选择（基于数据集分析）
  - standard: 标准YOLOv11-s训练
  - small_object: 小目标优化训练
  - marine: 海洋环境优化训练
  - two_stage: 两阶段渐进式训练
  - three_stage: 三阶段渐进式训练
  - synergistic: 协同优化训练
        """
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "standard", "small_object", "marine",
                 "two_stage", "three_stage", "synergistic"],
        help="训练策略 (默认: auto)"
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="仅准备数据集，不训练"
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="仅分析数据集，不训练"
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过数据集准备"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="自定义预训练模型路径"
    )

    args = parser.parse_args()


    # 仅分析模式
    if args.analyze_only:
        print("📊 数据集分析模式")
        if not check_environment():
            return 1

        strategy = analyze_marine_dataset(Config)
        print(f"\n🎯 推荐训练策略: {strategy}")
        return 0

    # 检查环境
    if not check_environment():
        print("\n是否继续? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            return 1

    # 准备数据集
    if not args.skip_prepare:
        print("\n📁 准备数据集...")
        train_count, val_count = prepare_ship_dataset()

        if train_count == 0:
            print("❌ 数据集准备失败，没有找到训练样本")
            return 1

        print(f"✅ 数据集准备完成: {train_count} 训练样本, {val_count} 验证样本")

        if args.prepare_only:
            return 0
    else:
        print("⏭️  跳过数据集准备")

    # 训练
    print(f"\n🚀 开始训练 [策略: {args.strategy}]")
    print("-" * 50)

    try:
        trainer = ShipDetectionTrainer(model_path=args.model_path)
        model, results = trainer.train(args.strategy)

        print("\n" + "=" * 50)
        print("✅ 训练完成！")
        print(f"📁 模型保存位置: runs/detect/*/weights/best.pt")
        print("=" * 50)

        return 0

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
