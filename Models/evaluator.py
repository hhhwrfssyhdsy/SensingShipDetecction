import torch
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
import numpy as np

class ProgressiveModelEvaluator:
    """
    渐进式模型评估器 - 集成不同阶段的检测结果
    """
    def __init__(self, model_paths, weights=None):
        self.models = [YOLO(path) for path in model_paths]
        self.weights = weights or [1.0] * len(model_paths)
        
    def ensemble_predict(self, image_path, iou_threshold=0.5, skip_box_thr=0.0001):
        """
        集成不同阶段模型的预测结果
        """
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # 获取每个模型的预测
        for model, weight in zip(self.models, self.weights):
            results = model(image_path, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy() * weight  # 应用权重
                    labels = result.boxes.cls.cpu().numpy()
                    
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)
        
        # 使用加权框融合
        if all_boxes:
            boxes, scores, labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels,
                iou_thr=iou_threshold,
                skip_box_thr=skip_box_thr
            )
            return boxes, scores, labels
        else:
            return np.array([]), np.array([]), np.array([])
    
    def evaluate_progressive_improvement(self, val_loader):
        """
        评估渐进式改进效果
        """
        print("📊 评估渐进式改进效果...")
        
        metrics = {}
        
        for i, model in enumerate(self.models):
            print(f"评估阶段 {i+1} 模型...")
            # 这里可以添加具体的评估逻辑
            # 比如计算每个阶段在大小目标上的mAP
            pass
            
        return metrics