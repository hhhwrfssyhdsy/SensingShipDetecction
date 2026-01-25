import torch
import numpy as np
from Data.dataset import YOLODataset
from Data.transformer import get_marine_mosaic_transforms
from PIL import Image
import albumentations as A
import cv2

class MarineSmallObjectDataset(YOLODataset):
    """
    海洋环境小目标优化数据集
    """
    def __init__(self, img_paths, label_paths, transform=None, target_size=640, 
                 use_marine_mosaic=True, marine_strength=0.7):
        super().__init__(img_paths, label_paths, transform, target_size)
        self.use_marine_mosaic = use_marine_mosaic
        self.marine_mosaic_transform = get_marine_mosaic_transforms(target_size)
        self.marine_strength = marine_strength  # 海洋增强强度
        
    def __getitem__(self, idx):
        # 根据海洋增强强度决定是否使用Mosaic
        use_mosaic = self.use_marine_mosaic and (torch.rand(1) < self.marine_strength)
        
        if use_mosaic:
            return self._marine_mosaic_augmentation(idx)
        else:
            return super().__getitem__(idx)
    
    def _marine_mosaic_augmentation(self, idx):
        """
        海洋环境Mosaic增强 - 保持海洋特性
        """
        indices = [idx] + [torch.randint(0, len(self), (1,)).item() for _ in range(3)]
        
        mosaic_images = []
        mosaic_boxes = []
        mosaic_labels = []
        
        # 创建马赛克画布
        mosaic_img = np.zeros((self.target_size * 2, self.target_size * 2, 3), dtype=np.uint8)
        
        # 四个区域的位置
        positions = [
            (0, 0),  # 左上
            (self.target_size, 0),  # 右上  
            (0, self.target_size),  # 左下
            (self.target_size, self.target_size)  # 右下
        ]
        
        for i, pos_idx in enumerate(indices):
            # 读取原始图像和标签
            img_path = self.img_paths[pos_idx]
            image = np.array(Image.open(img_path).convert('RGB'))
            
            label_path = self.label_paths[pos_idx]
            original_size = (image.shape[1], image.shape[0])
            boxes, labels = self.parse_yolo_label(label_path, original_size)
            
            # 调整图像大小到一半尺寸
            resized_img = cv2.resize(image, (self.target_size, self.target_size))
            
            # 应用基础海洋增强（在拼接前）
            if len(boxes) > 0 and torch.rand(1) < 0.5:
                # 随机应用一些海洋增强到单个图像
                augmented = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
                ])(image=resized_img)
                resized_img = augmented['image']
            
            # 放置到马赛克画布中
            x_offset, y_offset = positions[i]
            mosaic_img[y_offset:y_offset+self.target_size, 
                      x_offset:x_offset+self.target_size] = resized_img
            
            # 调整边界框坐标
            if len(boxes) > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.target_size / original_size[0]) + x_offset
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.target_size / original_size[1]) + y_offset
                mosaic_boxes.extend(boxes)
                mosaic_labels.extend(labels)
        
        # 应用海洋环境Mosaic变换
        transformed = self.marine_mosaic_transform(
            image=mosaic_img,
            bboxes=mosaic_boxes,
            labels=mosaic_labels
        )
        
        image = transformed['image']
        boxes = torch.FloatTensor(transformed['bboxes']) if transformed['bboxes'] else torch.zeros((0, 4))
        labels = torch.LongTensor(transformed['labels']) if transformed['labels'] else torch.zeros((0,))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,)),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }
        
        return image, target

    def analyze_marine_small_objects(self):
        """
        分析海洋环境中的小目标分布
        """
        small_obj_stats = {
            'total': 0,
            'small': 0,
            'medium': 0, 
            'large': 0,
            'avg_size': 0,
            'marine_difficulty': 0  # 海洋环境难度评分
        }
        
        size_distribution = []
        
        for i in range(min(len(self), 1000)):  # 抽样分析
            img_path = self.img_paths[i]
            image = Image.open(img_path)
            original_size = image.size
            
            label_path = self.label_paths[i]
            boxes, labels = self.parse_yolo_label(label_path, original_size)
            
            for box in boxes:
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height
                total_area = original_size[0] * original_size[1]
                relative_area = area / total_area
                
                # 目标尺寸分类
                if relative_area < 0.005:
                    small_obj_stats['small'] += 1
                elif relative_area < 0.02:
                    small_obj_stats['medium'] += 1
                else:
                    small_obj_stats['large'] += 1
                
                small_obj_stats['total'] += 1
                size_distribution.append(relative_area)
        
        if small_obj_stats['total'] > 0:
            small_obj_stats['avg_size'] = np.mean(size_distribution)
            small_ratio = small_obj_stats['small'] / small_obj_stats['total']
            medium_ratio = small_obj_stats['medium'] / small_obj_stats['total']
            
            # 计算海洋环境难度（小目标越多，难度越高）
            small_obj_stats['marine_difficulty'] = small_ratio * 0.7 + medium_ratio * 0.3
        
        print(f"海洋目标分析:")
        print(f"  小目标比例: {small_obj_stats['small']/small_obj_stats['total']:.2%}")
        print(f"  中目标比例: {small_obj_stats['medium']/small_obj_stats['total']:.2%}") 
        print(f"  大目标比例: {small_obj_stats['large']/small_obj_stats['total']:.2%}")
        print(f"  平均相对尺寸: {small_obj_stats['avg_size']:.4f}")
        print(f"  海洋检测难度: {small_obj_stats['marine_difficulty']:.2f}")
        
        return small_obj_stats