import glob
from Models.evaluator import ProgressiveModelEvaluator

def ensemble_inference(image_path):
    """
    使用渐进式训练的多个模型进行集成推理
    """
    # 获取所有阶段的模型
    model_paths = [
        'runs/detect/stage1_normal/weights/best.pt',
        'runs/detect/stage2_balanced/weights/best.pt', 
        'runs/detect/stage3_small_objects/weights/best.pt'
    ]
    
    # 创建评估器（可以调整权重）
    evaluator = ProgressiveModelEvaluator(
        model_paths, 
        weights=[0.3, 0.3, 0.4]  # 给小目标阶段更高权重
    )
    
    # 集成预测
    boxes, scores, labels = evaluator.ensemble_predict(image_path)
    
    return boxes, scores, labels

# 使用示例
if __name__ == "__main__":
    result = ensemble_inference("test_image.jpg")
    print(f"检测到 {len(result[0])} 个目标")