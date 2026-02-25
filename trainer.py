"""
训练器模块
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from ultralytics import YOLO

from Config.config import Config
from models.custom_yolo import create_baseline_model, create_improved_model, apply_attention_to_model


class ShipDetectionTrainer:
    """舰船检测训练器"""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or "./Config/ship_detection.yaml"
        self.output_root = Config.OUTPUT_ROOT
        self.training_history = {}

    def check_baseline_trained(self) -> Tuple[bool, Optional[Path]]:
        """
        检查Baseline模型是否已训练

        Returns:
            (是否已训练, 模型路径)
        """
        baseline_dir = self.output_root / "baseline" / "baseline"
        best_model_path = baseline_dir / "weights" / "best.pt"
        results_file =  self.output_root / "baseline" / "baseline_results.json"

        if best_model_path.exists() and results_file.exists():
            # 检查results.json是否包含有效结果
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    if results.get('metrics', {}).get('mAP50', 0) > 0:
                        return True, best_model_path
            except:
                pass
        return False, None

    def check_improved_trained(self) -> Tuple[bool, Optional[Path]]:
        """
        检查改进模型是否已训练完成

        Returns:
            (是否已训练, 模型路径)
        """
        improved_dir = self.output_root / "improved"
        best_model_path = improved_dir / "stage3_refinement" / "weights" / "best.pt"
        history_file = self.output_root / "training_history.json"

        if best_model_path.exists() and history_file.exists():
            # 检查training_history.json是否包含所有阶段
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    # 检查是否有3个阶段的结果
                    if all(f"stage_{i}" in history for i in range(1, 4)):
                        return True, best_model_path
            except:
                pass
        return False, None

    def visualize_results(self, baseline_metrics: Dict = None, improved_metrics: Dict = None):
        """
        可视化训练结果

        Args:
            baseline_metrics: Baseline模型评估指标（可选）
            improved_metrics: 改进模型评估指标（可选）
        """
        try:
            from evaluation.visualizer import ResultVisualizer, plot_training_curves, plot_stage_comparison
            from evaluation.evaluator import ModelEvaluator

            # 如果没有提供指标，尝试加载已保存的结果
            results = {}

            if baseline_metrics:
                results["Baseline_YOLO11s"] = baseline_metrics
            else:
                # 尝试加载Baseline结果
                baseline_results_file = self.output_root / "results" / "Baseline_YOLO11s_results.json"
                if baseline_results_file.exists():
                    with open(baseline_results_file, 'r') as f:
                        results["Baseline_YOLO11s"] = json.load(f)

            if improved_metrics:
                results["Improved_MarineYOLO"] = improved_metrics
            else:
                # 尝试加载改进模型结果
                improved_results_file = self.output_root / "results" / "Improved_MarineYOLO_results.json"
                if improved_results_file.exists():
                    with open(improved_results_file, 'r') as f:
                        results["Improved_MarineYOLO"] = json.load(f)

            if len(results) < 2:
                print("   ⚠️  缺少评估结果，跳过可视化")
                print("   请先运行模型评估")
                return

            # 创建可视化器
            visualizer = ResultVisualizer(output_dir="training_visualization")

            # 生成对比图表
            visualizer.plot_comparison_bar(results)
            visualizer.plot_radar(results)

            # 绘制训练曲线（如果存在results.csv）
            baseline_results_dir = self.output_root / "baseline" / "baseline"
            if baseline_results_dir.exists():
                plot_training_curves(baseline_results_dir, self.output_root / "training_visualization" / "baseline_training_curves.png")

            improved_results_dir = self.output_root / "improved" / "stage3_refinement"
            if improved_results_dir.exists():
                plot_training_curves(improved_results_dir, self.output_root / "training_visualization" / "improved_training_curves.png")

            # 绘制阶段对比图（如果有训练历史）
            if self.training_history:
                plot_stage_comparison(self.training_history, self.output_root / "training_visualization" / "stage_comparison.png")

            # 如果有两个模型的结果，计算并显示改进幅度
            if "Baseline_YOLO11s" in results and "Improved_MarineYOLO" in results:
                comparison = {}
                for key in ['mAP50', 'mAP50_95', 'precision', 'recall']:
                    baseline_val = results["Baseline_YOLO11s"].get(key, 0)
                    improved_val = results["Improved_MarineYOLO"].get(key, 0)

                    comparison[key] = {
                        'baseline': baseline_val,
                        'improved': improved_val,
                        'absolute_improvement': improved_val - baseline_val,
                        'relative_improvement_percent': (
                            (improved_val - baseline_val) / baseline_val * 100
                            if baseline_val > 0 else 0
                        )
                    }

                visualizer.plot_improvement(comparison)

                # 打印改进幅度
                print("\n   📈 改进幅度:")
                for key in ['mAP50', 'mAP50_95', 'precision', 'recall']:
                    if key in comparison:
                        improvement = comparison[key]['relative_improvement_percent']
                        print(f"      {key}: {improvement:+.2f}%")


        except ImportError as e:
            print(f"   ⚠️  缺少可视化依赖: {e}")
            print("   请安装: pip install matplotlib numpy")
        except Exception as e:
            print(f"   ⚠️  可视化生成失败: {e}")

    def check_stage_trained(self, stage_num: int) -> Tuple[bool, Optional[Path]]:
        """
        检查指定阶段是否已完成训练

        Args:
            stage_num: 阶段编号 (1, 2, 3)

        Returns:
            (是否已完成, 模型路径)
        """
        if stage_num < 1 or stage_num > len(Config.PROGRESSIVE_STAGES):
            return False, None

        stage_config = Config.PROGRESSIVE_STAGES[stage_num - 1]
        stage_name = stage_config['name']

        output_dir = self.output_root / "improved"
        best_model_path = output_dir / stage_name / "weights" / "best.pt"
        results_file = output_dir / stage_name / f"{stage_name}_results.json"

        # 检查模型文件和结果文件是否存在
        if best_model_path.exists() and results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    # 检查是否有有效指标
                    metrics = results.get('metrics', {})
                    if metrics.get('mAP50', 0) > 0 or metrics.get('completed', False):
                        return True, best_model_path
            except:
                pass

        return False, None

    def verify_improvement_applied(self, model, expected_attention: str) -> bool:
        """
        验证改进模型是否成功应用了注意力机制

        Args:
            model: 模型实例
            expected_attention: 期望的注意力类型

        Returns:
            是否成功应用
        """
        # 检查1: 模型是否为ImprovedYOLO类型
        if hasattr(model, 'hook_manager') and hasattr(model, 'current_attention'):
            if model.current_attention == expected_attention:
                # 检查是否有有效的hook
                if hasattr(model.hook_manager, 'hooks') and len(model.hook_manager.hooks) > 0:
                    print(f"   ✅ 改进模型验证通过: {expected_attention.upper()} 注意力已应用")
                    print(f"   已注册 {len(model.hook_manager.hooks)} 个Hook")
                    return True
                else:
                    print(f"   ⚠️  改进模型警告: Hook管理器存在但没有活跃的Hook")
                    return False
            else:
                print(f"   ⚠️  改进模型警告: 期望 {expected_attention.upper()}，但当前为 {model.current_attention}")
                return False

        # 检查2: 模型是否有附加的hook_manager（通过apply_attention_to_model添加）
        if hasattr(model, '_hook_manager'):
            hook_manager = model._hook_manager
            if hasattr(hook_manager, 'hooks') and len(hook_manager.hooks) > 0:
                print(f"   ✅ 改进模型验证通过: 注意力机制已应用 ({len(hook_manager.hooks)} 个Hook)")
                return True

        print(f"   ❌ 改进模型验证失败: 未检测到注意力机制")
        print(f"   模型类型: {type(model).__name__}")
        return False

    def train_baseline(self, skip_if_trained: bool = True) -> Tuple[YOLO, Dict]:
        """
        训练Baseline模型

        Args:
            skip_if_trained: 如果模型已训练是否跳过

        Returns:
            (模型, 训练结果)
        """
        print("\n" + "=" * 60)
        print("🚀 训练 Baseline 模型 (YOLO11s)")
        print("=" * 60)

        # 检查是否已训练
        if skip_if_trained:
            is_trained, model_path = self.check_baseline_trained()
            if is_trained:
                print(f"   ✅ Baseline 模型已训练，跳过训练")
                print(f"   模型路径: {model_path}")

                # 加载已训练的模型
                model = YOLO(str(model_path))

                # 读取已保存的结果
                results_file = self.output_root / "baseline" / "baseline_results.json"
                with open(results_file, 'r') as f:
                    saved_results = json.load(f)

                print(f"   历史 mAP50: {saved_results.get('metrics', {}).get('mAP50', 0):.4f}")
                return model, saved_results

        print(f"   总轮数: {Config.BASELINE_EPOCHS}")

        # 创建输出目录
        output_dir = self.output_root / "baseline"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型
        model = create_baseline_model(Config.BASELINE_MODEL)

        # 训练
        results = model.train(
            data=self.data_path,
            epochs=Config.BASELINE_EPOCHS,
            imgsz=Config.TARGET_SIZE,
            batch=Config.BATCH_SIZE,
            device=Config.DEVICE,
            workers=Config.NUM_WORKERS,
            project=str(output_dir),
            name="baseline",
            task='obb',
            verbose=True,
            exist_ok=True,
        )

        # 保存结果
        self._save_training_results("baseline", results, output_dir)

        # 提取并显示指标
        metrics = self._extract_metrics(results)
        print(f"\n✅ Baseline 训练完成")
        print(f"   最佳 mAP50: {metrics['mAP50']:.4f}")
        print(f"   最佳 mAP50-95: {metrics['mAP50_95']:.4f}")

        return model, results

    def train_progressive(self, skip_if_trained: bool = True) -> Tuple[YOLO, Dict]:
        """
        三阶段渐进训练
        支持断点续训：自动检测已完成的阶段并跳过
        支持改进验证：确保注意力机制成功应用

        Args:
            skip_if_trained: 如果模型已训练是否跳过

        Returns:
            (最终模型, 训练历史)
        """
        print("\n" + "=" * 60)
        print("🚀 训练改进模型 (三阶段渐进训练)")
        print("=" * 60)

        stages = Config.PROGRESSIVE_STAGES
        output_dir = self.output_root / "improved"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 检查各阶段完成状态
        print("\n📋 检查阶段训练状态:")
        stage_status = {}
        for i in range(1, len(stages) + 1):
            is_complete, model_path = self.check_stage_trained(i)
            stage_status[i] = {
                'complete': is_complete,
                'path': model_path
            }
            status = "✅ 已完成" if is_complete else "⏳ 待训练"
            print(f"   Stage {i}: {status}")

        # 检查是否全部完成
        if skip_if_trained and all(s['complete'] for s in stage_status.values()):
            print(f"\n   ✅ 所有阶段已训练完成，跳过训练")
            final_model_path = stage_status[3]['path']
            print(f"   最终模型: {final_model_path}")

            # 加载最终模型
            model = YOLO(str(final_model_path))

            # 读取训练历史
            history_file = self.output_root / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.training_history = json.load(f)

            # 显示各阶段结果
            for i in range(1, 4):
                stage_key = f"stage_{i}"
                if stage_key in self.training_history:
                    metrics = self.training_history[stage_key].get('metrics', {})
                    print(f"   Stage {i} mAP50: {metrics.get('mAP50', 0):.4f}")

            return model, self.training_history

        # 计算剩余训练轮数
        remaining_epochs = sum(
            stages[i-1]['epochs'] for i in range(1, len(stages) + 1)
            if not stage_status[i]['complete']
        )
        print(f"\n   剩余轮数: {remaining_epochs}")
        print(f"   阶段数: {len(stages)}")

        model = None
        best_model_path = None

        for i, stage in enumerate(stages, 1):
            # 检查该阶段是否已完成
            if stage_status[i]['complete']:
                print(f"\n{'=' * 60}")
                print(f"📌 Stage {i}: {stage['description']}")
                print(f"{'=' * 60}")
                print(f"   ✅ 该阶段已训练完成，跳过")
                best_model_path = stage_status[i]['path']

                # 加载该阶段模型用于下一阶段
                if i < len(stages):
                    model = YOLO(str(best_model_path))

                # 加载该阶段的历史记录
                stage_results_file = output_dir / stage['name'] / f"{stage['name']}_results.json"
                if stage_results_file.exists():
                    with open(stage_results_file, 'r') as f:
                        stage_results = json.load(f)
                    self.training_history[f"stage_{i}"] = {
                        "config": stage,
                        "metrics": stage_results.get('metrics', {})
                    }
                continue

            print(f"\n{'=' * 60}")
            print(f"📌 Stage {i}: {stage['description']}")
            print(f"   轮数: {stage['epochs']}")
            print(f"   注意力: {stage['attention']}")
            print(f"   Mosaic: {stage.get('mosaic', 0.0)}")
            print(f"   Mixup: {stage.get('mixup', 0.0)}")
            print(f"   Box Loss: {stage.get('box', 7.5)}")
            print(f"{'=' * 60}")

            # 创建或加载模型
            attention_applied = False
            if i == 1:
                model, attention_applied = create_improved_model(stage['attention'])
            else:
                # 加载上一阶段最佳模型
                if best_model_path and Path(best_model_path).exists():
                    print(f"   加载上一阶段模型: {best_model_path}")
                    model = YOLO(best_model_path)
                    # 为加载的模型应用新的注意力
                    model, attention_applied = apply_attention_to_model(model, stage['attention'])
                else:
                    model, attention_applied = create_improved_model(stage['attention'])

            # ========== 功能1: 验证改进模型是否成功应用 ==========
            print("\n   🔍 验证改进模型...")

            # 首先检查创建/应用时的返回值
            if not attention_applied:
                print("\n" + "=" * 60)
                print("❌ 错误: 改进模型创建失败")
                print("   注意力机制未能成功应用到模型")
                print("   训练已停止，请检查模型配置")
                print("=" * 60)
                raise RuntimeError("改进模型验证失败: 注意力机制未成功应用")

            # 进一步验证Hook是否正确注册
            improvement_verified = self.verify_improvement_applied(model, stage['attention'])
            if not improvement_verified:
                print("\n" + "=" * 60)
                print("❌ 错误: 改进模型验证失败")
                print("   注意力机制未能成功应用到模型")
                print("   训练已停止，请检查模型配置")
                print("=" * 60)
                raise RuntimeError("改进模型验证失败: 注意力机制未成功应用")

            # 训练参数
            train_args = {
                'data': self.data_path,
                'epochs': stage['epochs'],
                'imgsz': Config.TARGET_SIZE,
                'batch': Config.BATCH_SIZE,
                'device': Config.DEVICE,
                'workers': Config.NUM_WORKERS,
                'project': str(output_dir),
                'name': stage['name'],
                'task': 'obb',
                'verbose': True,
                'exist_ok': True,
                'lr0': stage.get('lr0', 0.01),
                'lrf': stage.get('lrf', 0.01),
                'mosaic': stage.get('mosaic', 0.0),
                'mixup': stage.get('mixup', 0.0),
                'box': stage.get('box', 7.5),
            }

            # 添加可选参数
            if 'copy_paste' in stage:
                train_args['copy_paste'] = stage['copy_paste']

            # 训练
            results = model.train(**train_args)

            # 保存阶段结果
            self._save_stage_results(i, stage, results, output_dir)

            # 更新最佳模型路径
            best_model_path = output_dir / stage['name'] / "weights" / "best.pt"

            # 提取并显示指标
            stage_metrics = self._extract_metrics(results)
            print(f"\n✅ Stage {i} 完成")
            print(f"   mAP50: {stage_metrics['mAP50']:.4f}")
            print(f"   mAP50-95: {stage_metrics['mAP50_95']:.4f}")

        # 保存完整训练历史
        self._save_training_history()

        # 加载最终模型
        final_model = YOLO(best_model_path) if best_model_path and best_model_path.exists() else model

        print(f"\n✅ 渐进训练完成")
        print(f"   最终模型: {best_model_path}")

        return final_model, self.training_history

    def _save_stage_results(self, stage_num: int, stage_config: Dict, results, output_dir: Path = None):
        """保存阶段训练结果"""
        metrics = self._extract_metrics(results)

        # 保存到训练历史
        self.training_history[f"stage_{stage_num}"] = {
            "config": stage_config,
            "metrics": metrics,
            "completed": True  # 标记该阶段已完成
        }

        # 同时保存到独立文件（用于断点续训检查）
        if output_dir:
            stage_name = stage_config['name']
            results_dict = {
                "stage": stage_num,
                "stage_name": stage_name,
                "metrics": metrics,
                "completed": True,
                "timestamp": str(datetime.now())
            }

            results_file = output_dir / stage_name / f"{stage_name}_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)

    def _extract_metrics(self, results) -> Dict[str, float]:
        """
        从训练结果中提取评估指标

        Args:
            results: Ultralytics训练结果

        Returns:
            指标字典
        """
        metrics = {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        # 尝试不同的键名格式（包括OBB任务的键名）
        if hasattr(results, 'results_dict'):
            rd = results.results_dict

            # mAP50 - 尝试多种可能的键名
            map50_keys = [
                'metrics/mAP50', 'metrics/mAP50(B)', 'metrics/mAP50(M)',
                'metrics/mAP50(OBB)', 'mAP50', 'mAP50(B)', 'mAP50(M)',
                'metrics/mAP50-95(OBB)',  # OBB任务可能使用这个键
            ]
            for key in map50_keys:
                if key in rd and rd[key] is not None:
                    metrics["mAP50"] = float(rd[key])
                    break

            # mAP50-95
            map5095_keys = [
                'metrics/mAP50-95', 'metrics/mAP50-95(B)', 'metrics/mAP50-95(M)',
                'metrics/mAP50-95(OBB)', 'mAP50-95', 'mAP50-95(B)', 'mAP50-95(M)'
            ]
            for key in map5095_keys:
                if key in rd and rd[key] is not None:
                    metrics["mAP50_95"] = float(rd[key])
                    break

            # Precision
            precision_keys = [
                'metrics/precision(B)', 'metrics/precision', 'precision',
                'metrics/precision(M)', 'precision(M)'
            ]
            for key in precision_keys:
                if key in rd and rd[key] is not None:
                    metrics["precision"] = float(rd[key])
                    break

            # Recall
            recall_keys = [
                'metrics/recall(B)', 'metrics/recall', 'recall',
                'metrics/recall(M)', 'recall(M)'
            ]
            for key in recall_keys:
                if key in rd and rd[key] is not None:
                    metrics["recall"] = float(rd[key])
                    break

        # 如果results_dict中没有，尝试从results对象直接获取
        if metrics["mAP50"] == 0.0:
            # 尝试不同的属性名
            if hasattr(results, 'box'):
                box = results.box
                if hasattr(box, 'map50') and box.map50 is not None:
                    metrics["mAP50"] = float(box.map50)
                if hasattr(box, 'map') and box.map is not None:
                    metrics["mAP50_95"] = float(box.map)
            elif hasattr(results, 'obb'):
                # OBB任务可能使用obb属性
                obb = results.obb
                if hasattr(obb, 'map50') and obb.map50 is not None:
                    metrics["mAP50"] = float(obb.map50)
                if hasattr(obb, 'map') and obb.map is not None:
                    metrics["mAP50_95"] = float(obb.map)

        return metrics

    def _save_training_results(self, name: str, results, output_dir: Path):
        """保存训练结果"""
        metrics = self._extract_metrics(results)

        results_dict = {
            "model": name,
            "metrics": metrics
        }

        output_file = output_dir / f"{name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"   训练结果已保存: {output_file}")
        print(f"   mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")

    def _save_training_history(self):
        """保存完整训练历史"""
        history_file = self.output_root / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\n   训练历史已保存: {history_file}")


    def export_for_edge_deployment(
        self,
        baseline_model_path: str = None,
        improved_model_path: str = None
    ) -> Dict[str, str]:
        """
        导出Baseline和改进模型用于边缘设备部署

        Args:
            baseline_model_path: Baseline模型路径，默认使用训练输出
            improved_model_path: 改进模型路径，默认使用训练输出

        Returns:
            导出的模型路径字典
        """
        from models.edge_optimization import export_comparison_models

        print("\n" + "=" * 60)
        print("📦 导出模型用于边缘设备部署")
        print("=" * 60)

        # 默认路径
        if baseline_model_path is None:
            baseline_model_path = str(self.output_root / "baseline" / "baseline" / "weights" / "best.pt")

        if improved_model_path is None:
            improved_model_path = str(self.output_root / "improved" / "stage3_refinement" / "weights" / "best.pt")

        # 检查模型是否存在
        baseline_exists = Path(baseline_model_path).exists()
        improved_exists = Path(improved_model_path).exists()

        if not baseline_exists:
            print(f"   ⚠️  Baseline模型不存在: {baseline_model_path}")
            print("      请先训练Baseline模型")
            return {}

        if not improved_exists:
            print(f"   ⚠️  改进模型不存在: {improved_model_path}")
            print("      请先训练改进模型")
            return {}

        # 导出模型
        output_dir = str(self.output_root / "edge_deployment")
        results = export_comparison_models(
            baseline_model_path,
            improved_model_path,
            output_dir
        )

        return results


def train_all(skip_trained: bool = True):
    """
    训练所有模型的便捷函数

    Args:
        skip_trained: 是否跳过已训练的模型
    """
    trainer = ShipDetectionTrainer()

    # 检查Baseline是否已训练
    baseline_trained, baseline_path = trainer.check_baseline_trained()
    if baseline_trained and skip_trained:
        print("\n" + "=" * 60)
        print("📋 训练状态检查")
        print("=" * 60)
        print(f"   ✅ Baseline 模型已训练")
        print(f"   路径: {baseline_path}")

    # 检查改进模型是否已训练
    improved_trained, improved_path = trainer.check_improved_trained()
    if improved_trained and skip_trained:
        print(f"   ✅ 改进模型已训练完成")
        print(f"   路径: {improved_path}")
        print(f"\n   两个模型都已训练，跳过训练阶段")
        print(f"   如需重新训练，请设置 skip_trained=False")
        print("=" * 60)

        # 加载已训练的模型
        baseline_model = YOLO(str(baseline_path))
        improved_model = YOLO(str(improved_path))

        # 导出边缘部署模型（如果需要）
        if Config.EDGE_OPTIMIZATION['enabled']:
            print("\n📦 导出边缘部署模型...")
            export_results = trainer.export_for_edge_deployment(
                str(baseline_path),
                str(improved_path)
            )
            if export_results:
                print("   ✅ 模型导出完成，可用于边缘设备部署")

        # 生成可视化报告（即使跳过训练也生成）
        trainer.visualize_results()

        return baseline_model, improved_model

    # 训练Baseline（如果未训练或skip_trained=False）
    baseline_model, _ = trainer.train_baseline(skip_if_trained=skip_trained)

    # 训练改进模型（如果未训练或skip_trained=False）
    improved_model, _ = trainer.train_progressive(skip_if_trained=skip_trained)

    # 导出边缘部署模型
    if Config.EDGE_OPTIMIZATION['enabled']:
        print("\n📦 导出边缘部署模型...")
        export_results = trainer.export_for_edge_deployment()
        if export_results:
            print("   ✅ 模型导出完成，可用于边缘设备部署")

    # 生成可视化报告
    trainer.visualize_results()

    return baseline_model, improved_model


def export_for_edge():
    """仅导出模型用于边缘部署"""
    trainer = ShipDetectionTrainer()
    return trainer.export_for_edge_deployment()


def visualize_training_results(baseline_metrics: Dict, improved_metrics: Dict, comparison: Dict = None):
    """
    可视化训练结果

    Args:
        baseline_metrics: Baseline模型评估指标
        improved_metrics: 改进模型评估指标
        comparison: 对比结果（可选）
    """
    try:
        from evaluation.visualizer import ResultVisualizer

        print("\n" + "=" * 60)
        print("📊 生成可视化图表")
        print("=" * 60)

        visualizer = ResultVisualizer()

        results = {
            "Baseline_YOLO11s": baseline_metrics,
            "Improved_MarineYOLO": improved_metrics
        }

        visualizer.generate_report(results, comparison)

        print("   ✅ 可视化图表生成完成")
        print(f"   保存位置: {Config.OUTPUT_ROOT / 'visualization'}")
        print("=" * 60)

    except Exception as e:
        print(f"   ⚠️  可视化生成失败: {e}")
        print("   请检查是否安装了 matplotlib 和 numpy")


def train_and_visualize(skip_trained: bool = True) -> Tuple[YOLO, YOLO]:
    """
    训练所有模型并生成可视化报告

    Args:
        skip_trained: 是否跳过已训练的模型

    Returns:
        (baseline_model, improved_model)
    """
    from evaluation.evaluator import ModelEvaluator

    # 训练模型
    baseline_model, improved_model = train_all(skip_trained=skip_trained)

    # 评估模型
    print("\n" + "=" * 60)
    print("📊 评估模型性能")
    print("=" * 60)

    data_path = "./Config/ship_detection.yaml"

    # 评估Baseline
    baseline_evaluator = ModelEvaluator(baseline_model, "Baseline_YOLO11s")
    baseline_metrics = baseline_evaluator.evaluate(data_path, save_results=True)

    # 评估改进模型
    improved_evaluator = ModelEvaluator(improved_model, "Improved_MarineYOLO")
    improved_metrics = improved_evaluator.evaluate(data_path, save_results=True)

    # 对比
    comparison = improved_evaluator.compare_with_baseline(baseline_model, data_path)

    # 打印对比结果
    print("\n📊 精度指标对比:")
    print(f"   {'Metric':<20} {'Baseline':<12} {'Improved':<12} {'Improvement':<12}")
    print("   " + "-" * 60)
    for key in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        if key in comparison:
            baseline_val = comparison[key]['baseline']
            improved_val = comparison[key]['improved']
            improvement = comparison[key]['relative_improvement_percent']
            print(f"   {key:<20} {baseline_val:<12.4f} {improved_val:<12.4f} {improvement:+.2f}%")

    # 生成可视化
    visualize_training_results(baseline_metrics, improved_metrics, comparison)

    return baseline_model, improved_model


if __name__ == "__main__":
    train_all()
