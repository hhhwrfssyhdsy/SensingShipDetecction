from ultralytics import YOLO
import torch
import yaml
import os
from Config.DataReading import *
from Models import attention_modules
from Config import DataReading
from Config.DataReading import (
        SSDD_train_inshore_img_path,
        SSDD_train_offshore_img_path, 
        SSDD_train_label_path,
        S_img_path,
        S_label_path,
        BATCH_SIZE,
        NUM_WORKERS,
        YOLO_model_path,
        TARGET_SIZE
    )

def create_dataset_config():
    """
    创建 YOLO 标准数据集配置文件
    """

    data_config = {
        # YOLO 数据集根目录
        'path': r"/root/ShipDetection_improved",

        # 相对 path 的子目录
        'train': 'images/train',
        'val': 'images/val',

        # 类别信息
        'nc': 1,
        'names': ['ship']
    }

    os.makedirs('./Config', exist_ok=True)

    yaml_path = './Config/ship_detection.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True)

    return yaml_path


def train_yolov11s_ship_detection():
    """
    使用官方预训练的YOLOv11-s模型进行船舶检测训练
    """
    # 创建数据集配置
    data_config_path = create_dataset_config()
    
    # 加载官方预训练的YOLOv11-s模型
    model = YOLO(YOLO_model_path)
    
    print("✅ 已加载官方预训练的YOLOv11-s模型")
    
    # 训练配置
    results = model.train(
        data=data_config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=15,
        save=True,
        pretrained=True,
        optimizer='auto',
        verbose=True
    )
    
    return model, results

def create_small_object_model_config():
    """
    创建小目标优化的模型配置
    """
    model_config = {
        'nc': 1,
        'scale': 's',
        'small_object_optimized': True,
    }
    return model_config

def train_yolov11s_ship_detection_small_object():
    """
    小目标优化的YOLOv11-s训练
    """
    data_config_path = create_dataset_config()
    model = YOLO(YOLO_model_path)
    
    print("✅ 已加载官方预训练的YOLOv11-s模型")
    print("🎯 启用小目标优化配置...")
    
    results = model.train(
        data=data_config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=20,
        save=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        lr0=0.01,
        lrf=0.01,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        anchor_t=3.0,
        fl_gamma=1.5,
    )
    return model, results

def train_marine_small_object_detection():
    """
    海洋环境小目标检测训练
    """
    data_config_path = create_dataset_config()
    model = YOLO(YOLO_model_path)
    
    print("🌊 启用海洋环境小目标优化训练...")
    
    results = model.train(
        data=data_config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=20,
        save=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        mosaic=0.7,
        mixup=0.15,
        copy_paste=0.1,
        box=7.5,
        cls=0.6,
        dfl=1.5,
        lr0=0.01,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        anchor_t=3.0,
        fl_gamma=1.5,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
    )
    return model, results

def train_progressive_ship_detection():
    """
    渐进式船舶检测训练 - 先正常训练，再小目标强化
    """
    data_config_path = create_dataset_config()
    
    print("🔄 开始渐进式训练...")
    
    # 阶段1: 标准训练（建立基础检测能力）
    print("🎯 阶段1: 标准训练 - 建立基础检测能力")
    model_stage1 = YOLO(YOLO_model_path)
    
    results_stage1 = model_stage1.train(
        data=data_config_path,
        epochs=80,  # 总epochs的60%
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=10,
        save=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        
        # 标准训练参数
        mosaic=0.3,
        mixup=0.05,
        copy_paste=0.02,
        box=5.0,
        cls=1.0,
        dfl=1.5,
        lr0=0.01,
        lrf=0.001,
        
        # 保存阶段1模型
        name='ship_detection_stage1',
    )
    
    # 阶段2: 小目标强化训练
    print("🔍 阶段2: 小目标强化训练")
    
    # 加载阶段1的最佳模型
    model_stage2 = YOLO('runs/detect/ship_detection_stage1/weights/best.pt')
    
    results_stage2 = model_stage2.train(
        data=data_config_path,
        epochs=50,  # 总epochs的40%
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=8,
        save=True,
        pretrained=False,  # 从阶段1继续训练
        optimizer='auto',
        verbose=True,
        
        # 小目标强化参数
        mosaic=0.7,
        mixup=0.15,
        copy_paste=0.1,
        box=7.5,
        cls=0.6,
        dfl=2.0,
        lr0=0.001,  # 更小的学习率进行微调
        lrf=0.0001,
        
        # 小目标专用增强
        hsv_h=0.01,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=8.0,  # 减少旋转避免小目标丢失
        translate=0.05,  # 减少平移
        scale=0.3,
        
        # 保存最终模型
        name='ship_detection_final',
    )
    
    return model_stage2, {'stage1': results_stage1, 'stage2': results_stage2}

def train_three_stage_progressive():
    """
    三阶段渐进式训练
    """
    data_config_path = create_dataset_config()
    
    print("🔄 开始三阶段渐进式训练...")
    
    # 阶段1: 基础训练（正常目标）
    print("🎯 阶段1: 基础训练 - 正常目标")
    model = YOLO('yolov11s.pt')
    
    stage1_results = model.train(
        data=data_config_path,
        epochs=60,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=8,
        save=True,
        pretrained=True,
        name='stage1_normal',
        # 保守的数据增强
        mosaic=0.2,
        mixup=0.02,
    )
    
    # 阶段2: 平衡训练（所有目标）
    print("⚖️ 阶段2: 平衡训练 - 所有目标")
    model = YOLO('runs/detect/stage1_normal/weights/best.pt')
    
    stage2_results = model.train(
        data=data_config_path,
        epochs=40,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=6,
        save=True,
        name='stage2_balanced',
        # 平衡的参数
        mosaic=0.5,
        mixup=0.1,
        box=6.0,
        cls=0.8,
        lr0=0.005,  # 降低学习率
    )
    
    # 阶段3: 小目标强化
    print("🔍 阶段3: 小目标强化")
    model = YOLO('runs/detect/stage2_balanced/weights/best.pt')
    
    stage3_results = model.train(
        data=data_config_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=5,
        save=True,
        name='stage3_small_objects',
        # 小目标强化参数
        mosaic=0.8,
        mixup=0.2,
        copy_paste=0.15,
        box=8.0,
        cls=0.5,
        lr0=0.001,  # 更小的学习率
    )
    
    return model, {
        'stage1': stage1_results,
        'stage2': stage2_results, 
        'stage3': stage3_results
    }
    
    def create_custom_model_config():
        custom_config = {
        'nc': 1,
        'scale': 's',
        'backbone': {
            'type': 'CSPDarknet',
            'depth_multiple': 0.33,
            'width_multiple': 0.50,
        },
        'neck': {
            'type': 'PAN',
            'depth_multiple': 0.33,
            'width_multiple': 0.50,
        },
        'head': {
            'type': 'Detect',
            'num_layers': 4,  # 4个检测头
        }
    }
    
    os.makedirs('./config', exist_ok=True)
    with open('./config/custom_yolov11s.yaml', 'w') as f:
        yaml.dump(custom_config, f)
    
    return './config/custom_yolov11s.yaml'

def train_attention_enhanced_progressive():
    """
    注意力机制增强的渐进式训练
    """
    data_config_path = create_dataset_config()
    
    print("🔍 开始注意力机制增强的渐进式训练...")
    
    # 阶段1: 基础训练 + 数据增强
    print("🎯 阶段1: 基础训练 + 数据增强")
    model = YOLO(YOLO_model_path)
    
    stage1_results = model.train(
        data=data_config_path,
        epochs=60,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=8,
        save=True,
        pretrained=True,
        name='stage1_enhanced',
        # 数据增强参数
        mosaic=0.3,
        mixup=0.05,
        copy_paste=0.02,
    )
    
    # 阶段2: 模型结构优化
    print("🏗️ 阶段2: 模型结构优化")
    model = YOLO('runs/detect/stage1_enhanced/weights/best.pt')
    
    stage2_results = model.train(
        data=data_config_path,
        epochs=40,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=6,
        save=True,
        name='stage2_model_enhanced',
        # 模型优化参数
        mosaic=0.5,
        mixup=0.1,
        lr0=0.005,
        # 增强小目标检测
        box=7.0,
        cls=0.7,
    )
    
    # 阶段3: 注意力机制专项
    print("🎯 阶段3: 注意力机制专项")
    model = YOLO('runs/detect/stage2_model_enhanced/weights/best.pt')
    
    stage3_results = model.train(
        data=data_config_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        patience=5,
        save=True,
        name='stage3_attention_enhanced',
        # 注意力优化参数
        mosaic=0.8,
        mixup=0.2,
        lr0=0.001,
        # 强化小目标检测
        box=8.0,
        cls=0.5,
        dfl=2.0,
    )
    
    return model, {
        'stage1': stage1_results,
        'stage2': stage2_results,
        'stage3': stage3_results
    }

def train_synergistic_approach():
    """
    协同优化训练 - 数据增强 + 模型结构 + 注意力机制
    """
    print("🚀 启动协同优化训练...")
    
    # 分析数据集
    from Data.small_object_dataset import MarineSmallObjectDataset
    analysis_dataset = MarineSmallObjectDataset(
        img_paths=[SSDD_train_inshore_img_path, SSDD_train_offshore_img_path, S_img_path],
        label_paths=[SSDD_train_label_path, SSDD_train_label_path, S_label_path],
        use_marine_mosaic=False
    )
    
    stats = analysis_dataset.analyze_marine_small_objects()
    small_ratio = stats['small'] / stats['total']
    
    # 根据小目标比例选择训练策略
    if small_ratio > 0.4:
        print("🎯 检测到高小目标比例，使用完整协同优化")
        return train_attention_enhanced_progressive()
    else:
        print("🎯 使用标准渐进式训练")
        return train_three_stage_progressive()