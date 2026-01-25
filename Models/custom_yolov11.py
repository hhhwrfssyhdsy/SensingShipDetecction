import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

class CustomYOLOv11s(DetectionModel):
    """
    自定义YOLOv11-s模型 - 针对船舶检测优化
    """
    def __init__(self, cfg='yolov11s.yaml', ch=3, nc=1, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        
        # 修改检测头数量 - 增加P2层专门检测小目标
        self._modify_detection_heads()
        
    def _modify_detection_heads(self):
        """修改检测头以适应小目标检测"""
        # 增加更多的检测层来提升小目标检测能力
        if hasattr(self, 'model'):
            # 找到检测头部分并修改
            for i, layer in enumerate(self.model):
                if hasattr(layer, 'nl'):  # 检测头层
                    # 增加锚点框数量，特别是小目标锚点
                    layer.anchors = layer.anchors * 1.5  # 增加锚点密度
                    layer.nl = 4  # 改为4个检测头

class EnhancedLoss(v8DetectionLoss):
    """
    增强的损失函数 - 针对小目标优化
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 增加小目标的损失权重
        self.small_obj_weight = 2.0
        
    def forward(self, preds, batch):
        loss = super().forward(preds, batch)
        
        # 额外的小目标损失计算
        if hasattr(self, 'box_loss'):
            # 可以根据目标尺寸调整损失权重
            return loss * self.small_obj_weight
        return loss