import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

def get_train_transforms(target_size=640):
    """
    训练时的数据增强变换 - 小目标与海洋环境双优化
    """
    # 基础几何变换（保持小目标）
    geometric_transforms = A.OneOf([
        A.NoOp(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # 减少位移，避免小目标移出
            scale_limit=0.1,   # 减少缩放，保持小目标尺寸
            rotate_limit=15,   # 减少旋转角度
            p=0.7
        ),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.5
        )
    ], p=0.8)
    
    # 海洋环境增强（保持海洋特性）
    marine_environment_transforms = A.OneOf([
        A.MotionBlur(blur_limit=(3, 7), p=0.4),  # 海浪运动模糊
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),  # 传感器噪声
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.08, p=0.2),  # 海雾
        A.ChannelShuffle(p=0.1),  # 色彩变化
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),  # ISO噪声
    ], p=0.6)
    
    # 色彩增强（不影响小目标识别）
    color_transforms = A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.25,  # 适当降低亮度变化
            contrast_limit=0.25, 
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,    # 减少色相变化
            sat_shift_limit=30,    # 保持饱和度增强
            val_shift_limit=20, 
            p=0.6
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # 增强对比度
    ], p=0.7)
    
    # 组合所有变换
    marine_small_object_transform = A.Compose([
        # 必须的预处理
        A.Resize(height=target_size, width=target_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        
        # 几何变换（小目标友好）
        geometric_transforms,
        
        # 海洋环境增强
        marine_environment_transforms,
        
        # 色彩增强
        color_transforms,
        
        # 小目标专用增强
        A.ImageCompression(quality_lower=80, quality_upper=95, p=0.2),  # 轻微压缩
        
        # 标准化
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
        
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.15,  # 适当降低但不过分
        min_area=12,          # 平衡小目标和误检
        min_width=6,
        min_height=6
    ))
    
    return marine_small_object_transform

def get_marine_mosaic_transforms(target_size=640):
    """
    海洋环境Mosaic增强 - 保持海洋特性同时提升小目标
    """
    mosaic_transform = A.Compose([
        A.Resize(height=target_size, width=target_size, p=1.0),
        
        # 海洋环境增强（Mosaic后应用）
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 6), p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        ], p=0.5),
        
        # 色彩增强
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1,
        min_area=8
    ))
    
    return mosaic_transform

def get_advanced_marine_transforms(target_size=640):
    """
    高级海洋环境增强 - 分阶段应用
    """
    # 第一阶段：基础增强
    stage1 = A.Compose([
        A.Resize(height=target_size, width=target_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
    ])
    
    # 第二阶段：海洋环境增强
    stage2 = A.Compose([
        # 海洋特效
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, p=0.2),
        ], p=0.7),
        
        # 光照变化
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=25, p=0.5),
    ])
    
    # 第三阶段：小目标优化
    stage3 = A.Compose([
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return stage1, stage2, stage3

def get_val_transforms(target_size=640):
    """
    验证时的变换 - 保持海洋图像特性
    """
    val_transform = A.Compose([
        A.Resize(height=target_size, width=target_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1
    ))
    
    return val_transform