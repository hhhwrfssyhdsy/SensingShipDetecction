import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from Config.DataReading import *
from Data.transformer import get_train_transforms, get_val_transforms

class YOLODataset(Dataset):
    def __init__(self, img_paths, label_paths, transform=None, target_size=640):
        """
        初始化YOLO数据集
        """
        self.img_paths = []
        self.label_paths = []
        self.transform = transform
        self.target_size = target_size
        
        # 处理多个路径输入
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        if isinstance(label_paths, str):
            label_paths = [label_paths]
        
        # 收集所有图像和标签文件路径
        for img_path, label_path in zip(img_paths, label_paths):
            if os.path.exists(img_path) and os.path.exists(label_path):
                img_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                for img_file in img_files:
                    name_without_ext = os.path.splitext(img_file)[0]
                    label_file = os.path.join(label_path, name_without_ext + '.txt')
                    
                    if os.path.exists(label_file):
                        self.img_paths.append(os.path.join(img_path, img_file))
                        self.label_paths.append(label_file)
        
        print(f"Loaded {len(self.img_paths)} samples")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 读取标签
        label_path = self.label_paths[idx]
        boxes, labels = self.parse_yolo_label(label_path, original_size)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((self.target_size, self.target_size)),
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        # 调整边界框到目标尺寸
        scale_x = self.target_size / original_size[0]
        scale_y = self.target_size / original_size[1]
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        # 转换为tensor
        boxes = torch.FloatTensor(boxes) if len(boxes) > 0 else torch.zeros((0, 4))
        labels = torch.LongTensor(labels) if len(labels) > 0 else torch.zeros((0,))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,)),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }
        
        return image, target
    
    def parse_yolo_label(self, label_path, image_size):
        """
        解析YOLO格式的标签文件
        """
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                data = line.strip().split()
                if len(data) >= 5:
                    class_id = int(data[0])
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])
                    
                    # 转换为绝对坐标
                    x_center_abs = x_center * image_size[0]
                    y_center_abs = y_center * image_size[1]
                    width_abs = width * image_size[0]
                    height_abs = height * image_size[1]
                    
                    # 计算边界框坐标
                    x_min = x_center_abs - width_abs / 2
                    y_min = y_center_abs - height_abs / 2
                    x_max = x_center_abs + width_abs / 2
                    y_max = y_center_abs + height_abs / 2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
        
        return np.array(boxes), np.array(labels)

def create_datasets(target_size=TARGET_SIZE):
    """
    创建训练和验证数据集
    """
    # 获取变换
    train_transform = get_train_transforms(target_size)
    val_transform = get_val_transforms(target_size)
    
    # 创建SSDD训练数据集
    ssdd_train_img_paths = [SSDD_train_inshore_img_path, SSDD_train_offshore_img_path]
    ssdd_train_label_paths = [SSDD_train_label_path, SSDD_train_label_path]
    
    ssdd_train_dataset = YOLODataset(
        img_paths=ssdd_train_img_paths,
        label_paths=ssdd_train_label_paths,
        transform=train_transform,
        target_size=target_size
    )
    
    # 创建SeaShip数据集
    seaship_dataset = YOLODataset(
        img_paths=S_img_path,
        label_paths=S_label_path,
        transform=train_transform,
        target_size=target_size
    )
    
    # 合并数据集
    combined_train_dataset = ConcatDataset([ssdd_train_dataset, seaship_dataset])
    
    # 创建验证数据集
    ssdd_val_img_paths = [SSDD_test_inshore_img_path, SSDD_test_offshore_img_path]
    ssdd_val_label_paths = [SSDD_test_inshore_label_path, SSDD_test_offshore_label_path]
    
    val_dataset = YOLODataset(
        img_paths=ssdd_val_img_paths,
        label_paths=ssdd_val_label_paths,
        transform=val_transform,
        target_size=target_size
    )
    
    return combined_train_dataset, val_dataset

def collate_fn(batch):
    """
    自定义批处理函数
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    return images, targets

def create_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, target_size=TARGET_SIZE):
    """
    创建训练和验证数据加载器
    """
    train_dataset, val_dataset = create_datasets(target_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader