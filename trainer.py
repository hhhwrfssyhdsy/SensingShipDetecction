"""
训练器模块 - 简化版
支持Baseline训练和渐进式改进训练
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from ultralytics import YOLO

from Config.config import Config
from models.custom_yolo import create_baseline_model, create_improved_model


class ShipDetectionTrainer:
    """舰船检测训练器"""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or "./Config/ship_detection.yaml"
        self.output_root = Config.OUTPUT_ROOT
        self.training_history = {}

    def train_baseline(self) -> Tuple[YOLO, Dict]:
        """
        训练Baseline模型

        Returns:
            (模型, 训练结果)
        """
        print("\n" + "=" * 60)
        print("🚀 训练 Baseline 模型 (YOLO11s)")
        print("=" * 60)
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

        print(f"\n✅ Baseline 训练完成")
        print(f"   最佳 mAP50: {results.results_dict.get('metrics/mAP50', 0):.4f}")

        return model, results

    def train_progressive(self) -> Tuple[YOLO, Dict]:
        """
        三阶段渐进训练

        Returns:
            (最终模型, 训练历史)
        """
        print("\n" + "=" * 60)
        print("🚀 训练改进模型 (三阶段渐进训练)")
        print("=" * 60)

        stages = Config.PROGRESSIVE_STAGES
        total_epochs = sum(s['epochs'] for s in stages)
        print(f"   总轮数: {total_epochs} (60+40+30)")
        print(f"   阶段数: {len(stages)}")

        output_dir = self.output_root / "improved"
        output_dir.mkdir(parents=True, exist_ok=True)

        model = None
        best_model_path = None

        for i, stage in enumerate(stages, 1):
            print(f"\n{'=' * 60}")
            print(f"📌 Stage {i}: {stage['description']}")
            print(f"   轮数: {stage['epochs']}")
            print(f"   注意力: {stage['attention']}")
            print(f"   Mosaic: {stage.get('mosaic', 0.0)}")
            print(f"   Mixup: {stage.get('mixup', 0.0)}")
            print(f"   Box Loss: {stage.get('box', 7.5)}")
            print(f"{'=' * 60}")

            # 创建或加载模型
            if i == 1:
                model = create_improved_model(stage['attention'])
            else:
                # 加载上一阶段最佳模型
                if best_model_path and Path(best_model_path).exists():
                    print(f"   加载上一阶段模型: {best_model_path}")
                    model = YOLO(best_model_path)
                else:
                    model = create_improved_model(stage['attention'])

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
            self._save_stage_results(i, stage, results)

            # 更新最佳模型路径
            best_model_path = output_dir / stage['name'] / "weights" / "best.pt"

            print(f"\n✅ Stage {i} 完成")
            print(f"   mAP50: {results.results_dict.get('metrics/mAP50', 0):.4f}")

        # 保存完整训练历史
        self._save_training_history()

        # 加载最终模型
        final_model = YOLO(best_model_path) if best_model_path.exists() else model

        print(f"\n✅ 渐进训练完成")
        print(f"   最终模型: {best_model_path}")

        return final_model, self.training_history

    def _save_stage_results(self, stage_num: int, stage_config: Dict, results):
        """保存阶段训练结果"""
        self.training_history[f"stage_{stage_num}"] = {
            "config": stage_config,
            "metrics": {
                "mAP50": float(results.results_dict.get('metrics/mAP50', 0)),
                "mAP50_95": float(results.results_dict.get('metrics/mAP50-95', 0)),
                "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                "recall": float(results.results_dict.get('metrics/recall(B)', 0)),
            }
        }

    def _save_training_results(self, name: str, results, output_dir: Path):
        """保存训练结果"""
        results_dict = {
            "model": name,
            "metrics": {
                "mAP50": float(results.results_dict.get('metrics/mAP50', 0)),
                "mAP50_95": float(results.results_dict.get('metrics/mAP50-95', 0)),
                "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                "recall": float(results.results_dict.get('metrics/recall(B)', 0)),
            }
        }

        output_file = output_dir / f"{name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def _save_training_history(self):
        """保存完整训练历史"""
        history_file = self.output_root / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\n   训练历史已保存: {history_file}")


    def train_edge_optimized(self) -> Tuple[YOLO, Dict]:
        """
        训练边缘优化模型

        Returns:
            (优化后的模型, 训练结果)
        """
        from models.edge_optimization import EdgeOptimizer, create_optimized_model

        print("\n" + "=" * 60)
        print("🚀 训练边缘优化模型")
        print("=" * 60)

        edge_config = Config.EDGE_OPTIMIZATION
        edge_training = Config.EDGE_TRAINING_CONFIG

        print(f"   轻量化模块: {edge_config['use_lightweight_blocks']}")
        print(f"   模型剪枝: {edge_config['use_pruning']} (比例: {edge_config['pruning_ratio']})")
        print(f"   量化: {edge_config['use_quantization']} ({edge_config['quantization_bits']}bit)")
        print(f"   训练轮数: {edge_training['epochs']}")
        print(f"   批次大小: {edge_training['batch_size']}")

        # 1. 首先训练基础模型
        print("\n📌 步骤1: 训练基础模型")
        model = create_improved_model("se")

        output_dir = self.output_root / "edge_optimized"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = model.train(
            data=self.data_path,
            epochs=edge_training['epochs'],
            imgsz=Config.TARGET_SIZE,
            batch=edge_training['batch_size'],
            device=Config.DEVICE,
            workers=Config.NUM_WORKERS,
            project=str(output_dir),
            name="base_training",
            task='obb',
            lr0=edge_training['learning_rate'],
            weight_decay=edge_training['weight_decay'],
            label_smoothing=edge_training['label_smoothing'],
            dropout=edge_training['dropout'],
            verbose=True,
            exist_ok=True,
        )

        # 2. 应用边缘优化
        print("\n📌 步骤2: 应用边缘优化")
        optimizer = EdgeOptimizer(model.model)

        optimized_model = optimizer.optimize_for_edge(
            use_lightweight_blocks=edge_config['use_lightweight_blocks'],
            use_pruning=edge_config['use_pruning'],
            use_quantization=edge_config['use_quantization'],
            pruning_ratio=edge_config['pruning_ratio']
        )

        # 3. 基准测试
        print("\n📌 步骤3: 性能基准测试")
        benchmark_results = optimizer.benchmark()

        # 4. 导出ONNX
        print("\n📌 步骤4: 导出ONNX模型")
        onnx_path = optimizer.export_to_onnx(str(output_dir / "edge_model.onnx"))

        # 5. 保存结果
        edge_results = {
            "base_metrics": {
                "mAP50": float(results.results_dict.get('metrics/mAP50', 0)),
                "mAP50_95": float(results.results_dict.get('metrics/mAP50-95', 0)),
            },
            "benchmark": benchmark_results,
            "optimization_config": edge_config,
        }

        with open(output_dir / "edge_results.json", 'w') as f:
            json.dump(edge_results, f, indent=2)

        print(f"\n✅ 边缘优化模型训练完成")
        print(f"   模型路径: {output_dir}/base_training/weights/best.pt")
        print(f"   ONNX路径: {onnx_path}")

        return model, edge_results


def train_all():
    """训练所有模型的便捷函数"""
    trainer = ShipDetectionTrainer()

    # 训练Baseline
    baseline_model, _ = trainer.train_baseline()

    # 训练改进模型
    improved_model, _ = trainer.train_progressive()

    # 训练边缘优化模型
    if Config.EDGE_OPTIMIZATION['enabled']:
        edge_model, _ = trainer.train_edge_optimized()

    return baseline_model, improved_model


def train_edge_only():
    """仅训练边缘优化模型"""
    trainer = ShipDetectionTrainer()
    return trainer.train_edge_optimized()


if __name__ == "__main__":
    train_all()
