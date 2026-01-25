from Data.dataset import create_dataloaders
from Models.trainer import (
    train_yolov11s_ship_detection, 
    train_marine_small_object_detection,
    train_progressive_ship_detection,
    train_three_stage_progressive,
    train_synergistic_approach  # 新增协同训练
)
from Config.DataReading import *
from Data.small_object_dataset import MarineSmallObjectDataset

def analyze_dataset_for_progressive():
    """
    为渐进式训练分析数据集
    """
    print("🔍 分析数据集以确定渐进式训练策略...")
    
    analysis_dataset = MarineSmallObjectDataset(
        img_paths=[SSDD_train_inshore_img_path, SSDD_train_offshore_img_path, S_img_path],
        label_paths=[SSDD_train_label_path, SSDD_train_label_path, S_label_path],
        use_marine_mosaic=False
    )
    
    stats = analysis_dataset.analyze_marine_small_objects()
    
    # 确定渐进式策略
    small_ratio = stats['small'] / stats['total']
    
    if small_ratio > 0.5:
        strategy = 'three_stage'  # 高小目标比例用三阶段
        print("🎯 检测到高小目标比例，推荐三阶段渐进训练")
    elif small_ratio > 0.25:
        strategy = 'two_stage'    # 中等小目标比例用两阶段
        print("🎯 检测到中等小目标比例，推荐两阶段渐进训练")
    else:
        strategy = 'standard'     # 低小目标比例用标准训练
        print("🎯 小目标比例正常，推荐标准训练")
    
    return {
        'strategy': strategy,
        'small_ratio': small_ratio,
        'stats': stats
    }

def main():
    """
    主训练流程 - 协同优化版
    """
    print("🚢 开始船舶目标检测训练流程（协同优化版）...")
    
    # 步骤0: 分析数据集确定策略
    analysis = analyze_dataset_for_progressive()
    
    # 步骤1: 创建数据加载器
    print("📊 创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_size=TARGET_SIZE
    )
    
    print(f"训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")
    
    # 步骤2: 根据分析选择训练策略
    if analysis['small_ratio'] > 0.4:
        print("🚀 检测到高小目标比例，启动协同优化训练...")
        print("📋 训练策略: 数据增强 + 模型结构 + 注意力机制 三重优化")
        model, results = train_synergistic_approach()
    elif analysis['strategy'] == 'three_stage':
        print("🔄 开始三阶段渐进式训练...")
        model, results = train_three_stage_progressive()
    elif analysis['strategy'] == 'two_stage':
        print("🔄 开始两阶段渐进式训练...")
        model, results = train_progressive_ship_detection()
    else:
        print("🎯 开始标准YOLOv11-s模型训练...")
        model, results = train_yolov11s_ship_detection()
    
    print("✅ 训练完成！")
    
    # 步骤3: 显示训练结果总结
    print("\n📊 训练结果总结:")
    print(f"📁 最终模型保存在: runs/detect/train/weights/best.pt")
    
    if analysis['strategy'] != 'standard':
        print("🔗 提示: 可以使用 ensemble_inference.py 进行模型集成推理")
        print("🔍 提示: 可以使用 evaluator.py 进行详细性能评估")
    
    # 显示数据集分析结果
    print(f"\n📈 数据集分析结果:")
    print(f"   小目标比例: {analysis['small_ratio']:.2%}")
    print(f"   中目标比例: {analysis['stats']['medium']/analysis['stats']['total']:.2%}")
    print(f"   大目标比例: {analysis['stats']['large']/analysis['stats']['total']:.2%}")
    print(f"   海洋检测难度: {analysis['stats']['marine_difficulty']:.2f}")

if __name__ == "__main__":
    main()