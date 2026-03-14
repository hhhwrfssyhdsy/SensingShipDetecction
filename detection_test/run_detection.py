"""
运行检测测试脚本
使用训练好的模型对测试图片进行检测
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import numpy as np


def convert_to_python_types(obj):
    """
    递归地将 NumPy 类型转换为 Python 原生类型
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


def run_detection(
    model_path: str,
    test_images_dir: str,
    output_dir: str = "results",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    save_txt: bool = True,
    save_json: bool = True,
    visualize: bool = True
):
    """
    运行检测测试

    Args:
        model_path: 模型路径 (.pt 文件)
        test_images_dir: 测试图片目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        save_txt: 是否保存检测结果为txt格式
        save_json: 是否保存检测结果为json格式
        visualize: 是否可视化检测结果
    """
    print("=" * 60)
    print("🚀 开始检测测试")
    print("=" * 60)

    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    model = YOLO(model_path)

    # 获取模型信息
    print(f"   模型类型: {model.task}")
    print(f"   模型类别数: {len(model.names)}")
    print(f"   类别名称: {model.names}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有测试图片
    test_dir = Path(test_images_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"\n❌ 未找到测试图片: {test_images_dir}")
        return

    print(f"\n📁 找到 {len(image_files)} 张测试图片")

    # 运行检测
    print(f"\n🔍 开始检测...")
    print(f"   置信度阈值: {conf_threshold}")
    print(f"   IoU阈值: {iou_threshold}")

    results_list = []
    total_detections = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n   [{idx}/{len(image_files)}] 处理: {image_path.name}")

        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"      ⚠️  无法读取图片，跳过")
            continue

        # 运行检测
        results = model(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True
        )

        # 处理结果
        detections = []
        for result in results:
            # 根据模型任务类型获取检测框
            # OBB 任务使用 result.obb，其他任务使用 result.boxes
            if model.task == 'obb' and hasattr(result, 'obb') and result.obb is not None:
                boxes = result.obb
            elif result.boxes is not None:
                boxes = result.boxes
            else:
                continue
            
            # 检查是否有检测框
            if len(boxes) == 0:
                continue
            
            # 批量获取所有检测框信息
            class_ids = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            
            # 根据任务类型获取坐标
            if model.task == 'obb' and hasattr(boxes, 'xywhr'):
                # OBB 格式: x_center, y_center, width, height, angle
                xywhr = boxes.xywhr.cpu().numpy()
                for i in range(len(class_ids)):
                    cls_id = int(class_ids[i])
                    conf = float(confidences[i])
                    x_center, y_center, width, height, angle = xywhr[i]
                    # 转换为 Python float 类型
                    bbox = [
                        float(x_center), 
                        float(y_center), 
                        float(width), 
                        float(height), 
                        float(angle)
                    ]
                    
                    detection = {
                        'class_id': cls_id,
                        'class_name': model.names[cls_id],
                        'confidence': conf,
                        'bbox': bbox
                    }
                    detections.append(detection)
                    total_detections += 1
            else:
                # 水平框格式: x1, y1, x2, y2
                xyxy = boxes.xyxy.cpu().numpy()
                for i in range(len(class_ids)):
                    cls_id = int(class_ids[i])
                    conf = float(confidences[i])
                    x1, y1, x2, y2 = xyxy[i]
                    # 转换为 Python float 类型
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    
                    detection = {
                        'class_id': cls_id,
                        'class_name': model.names[cls_id],
                        'confidence': conf,
                        'bbox': bbox
                    }
                    detections.append(detection)
                    total_detections += 1

        # 打印检测结果
        if len(detections) > 0:
            print(f"      ✅ 检测到 {len(detections)} 个目标")
        else:
            print(f"      ⚠️  未检测到目标")

        # 可视化结果
        if visualize:
            annotated_image = results[0].plot()
            output_image_path = output_path / f"result_{image_path.name}"
            cv2.imwrite(str(output_image_path), annotated_image)
            print(f"      🖼️  可视化结果已保存")

        # 保存检测结果
        if save_txt:
            txt_path = output_path / f"{image_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for det in detections:
                    cls_id = det['class_id']
                    bbox = det['bbox']
                    bbox_str = ' '.join([f'{x:.6f}' for x in bbox])
                    f.write(f"{cls_id} {bbox_str}\n")

        # 记录结果
        result_info = {
            'image': image_path.name,
            'detections': detections,
            'detection_count': len(detections)
        }
        results_list.append(result_info)

    # 保存JSON结果
    if save_json:
        json_path = output_path / "detection_results.json"
        summary = {
            'model_path': model_path,
            'test_time': datetime.now().isoformat(),
            'total_images': len(image_files),
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / len(image_files) if image_files else 0,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'results': results_list
        }
        
        # 转换为 Python 原生类型
        summary = convert_to_python_types(summary)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 检测完成")
    print("=" * 60)
    print(f"   总图片数: {len(image_files)}")
    print(f"   总检测数: {total_detections}")
    print(f"   平均每图检测数: {total_detections / len(image_files):.2f}")
    print(f"\n   结果保存位置: {output_path}")
    print(f"      - 可视化图片: result_*.jpg")
    print(f"      - 检测标签: *.txt")
    print(f"      - 结果汇总: detection_results.json")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='运行检测测试')
    parser.add_argument(
        '--model',
        type=str,
        default='best.pt',
        help='模型路径 (.pt 或 .onnx 文件)'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='test_imgs',
        help='测试图片目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='输出目录'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='置信度阈值'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU阈值'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='不保存可视化结果'
    )
    parser.add_argument(
        '--no-txt',
        action='store_true',
        help='不保存txt格式结果'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='不保存json格式结果'
    )

    args = parser.parse_args()

    # 查找模型文件
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"模型文件不存在: {args.model}")
        return

    # 检查测试图片目录
    images_path = Path(args.images)

    if not images_path.exists():
        print(f"测试图片目录不存在: {args.images}")
        return

    # 运行检测
    run_detection(
        model_path=str(model_path),
        test_images_dir=str(images_path),
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_txt=not args.no_txt,
        save_json=not args.no_json,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
