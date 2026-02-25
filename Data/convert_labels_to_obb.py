"""
标签格式转换工具
将 (class, x_center, y_center, width, height, angle) 格式
转换为 Ultralytics OBB 格式 (class, x1, y1, x2, y2, x3, y3, x4, y4)
"""
import math
from pathlib import Path
from typing import List, Tuple

from Config.config import Config


def xywha_to_xyxyxyxy(class_id: int, x_center: float, y_center: float, 
                       width: float, height: float, angle_deg: float) -> Tuple[int, float, float, float, float, float, float, float, float]:
    """
    将 (x_center, y_center, width, height, angle) 转换为四个角点 (x1,y1,x2,y2,x3,y3,x4,y4)
    
    Args:
        class_id: 类别ID
        x_center: 中心点 x (归一化 0-1)
        y_center: 中心点 y (归一化 0-1)
        width: 宽度 (归一化 0-1)
        height: 高度 (归一化 0-1)
        angle_deg: 旋转角度（度）
        
    Returns:
        (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
    """
    # 角度转弧度
    angle_rad = math.radians(angle_deg)
    
    # 半宽半高
    w2 = width / 2
    h2 = height / 2
    
    # 计算四个角点（未旋转前，相对于中心）
    # 顺序：左上、右上、右下、左下（顺时针）
    corners = [
        (-w2, -h2),  # 左上
        (w2, -h2),   # 右上
        (w2, h2),    # 右下
        (-w2, h2),   # 左下
    ]
    
    # 旋转并平移
    rotated_corners = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for dx, dy in corners:
        # 旋转
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        # 平移到中心
        x = x_center + rx
        y = y_center + ry
        rotated_corners.extend([x, y])
    
    return (class_id, *rotated_corners)


def convert_label_file(input_path: Path, output_path: Path):
    """
    转换单个标签文件
    
    Args:
        input_path: 输入标签文件路径
        output_path: 输出标签文件路径
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 6:
            # 6列格式：class x_center y_center width height angle
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            angle = float(parts[5])
            
            # 转换为 OBB 格式
            converted = xywha_to_xyxyxyxy(class_id, x_center, y_center, width, height, angle)
            converted_lines.append(f"{converted[0]} {converted[1]:.6f} {converted[2]:.6f} {converted[3]:.6f} {converted[4]:.6f} {converted[5]:.6f} {converted[6]:.6f} {converted[7]:.6f} {converted[8]:.6f}")
        elif len(parts) == 5:
            # 5列格式：class x_center y_center width height（水平框，角度为0）
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 转换为 OBB 格式（角度为0）
            converted = xywha_to_xyxyxyxy(class_id, x_center, y_center, width, height, 0.0)
            converted_lines.append(f"{converted[0]} {converted[1]:.6f} {converted[2]:.6f} {converted[3]:.6f} {converted[4]:.6f} {converted[5]:.6f} {converted[6]:.6f} {converted[7]:.6f} {converted[8]:.6f}")
        elif len(parts) == 9:
            # 已经是 OBB 格式，直接保留
            converted_lines.append(line.strip())
        else:
            print(f"    警告: 跳过格式不正确的行: {line.strip()}")
    
    # 写入输出文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(converted_lines) + '\n' if converted_lines else '')


def convert_dataset_labels():
    """转换整个数据集的标签"""
    
    dataset_root = Config.DATASET_ROOT
    
    # 定义 splits
    splits = ['train', 'val', 'test']
    
    total_converted = 0
    
    for split in splits:
        input_dir = dataset_root / "labels" / split
        
        if not input_dir.exists():
            print(f"   ⚠️  {split}: 标签目录不存在 - {input_dir}")
            continue
        
        label_files = list(input_dir.glob("*.txt"))
        
        if not label_files:
            print(f"   ⚠️  {split}: 未找到标签文件")
            continue
        
        print(f"   📝 {split}: 找到 {len(label_files)} 个标签文件")
        
        converted_count = 0
        for label_file in label_files:
            try:
                # 备份原文件
                backup_path = label_file.with_suffix('.txt.backup')
                if not backup_path.exists():
                    label_file.rename(backup_path)
                
                # 转换并保存
                convert_label_file(backup_path, label_file)
                converted_count += 1
                
            except Exception as e:
                print(f"    错误: 转换 {label_file.name} 失败: {e}")
        
        print(f"   ✅ {split}: 成功转换 {converted_count} 个文件")
        total_converted += converted_count
    
    print(f"\n✅ 总共转换 {total_converted} 个标签文件")



if __name__ == "__main__":
    convert_dataset_labels()
