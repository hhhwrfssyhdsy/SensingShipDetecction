# 检测测试使用说明

## 快速开始

### 1. 使用默认配置运行

```bash
python detection_test/run_detection.py
```

这将使用改进模型对 `detection_test/test_imgs` 下的所有图片进行检测。(test_imgs需要自行配置)

### 2. 指定模型

```bash
# 使用 Baseline 模型
python detection_test/run_detection.py --model output/baseline/baseline/weights/best.pt

# 使用改进模型
python detection_test/run_detection.py --model output/improved/stage3_refinement/weights/best.pt

# 使用 ONNX 模型
python detection_test/run_detection.py --model output/edge_deployment/baseline/baseline_yolo11s.onnx
```

### 3. 调整检测参数

```bash
# 提高置信度阈值（减少误检）
python detection_test/run_detection.py --conf 0.5

# 降低置信度阈值（提高召回率）
python detection_test/run_detection.py --conf 0.1

# 调整 IoU 阈值
python detection_test/run_detection.py --iou 0.5
```

### 4. 自定义输入输出

```bash
# 指定测试图片目录
python detection_test/run_detection.py --images path/to/test/images

# 指定输出目录
python detection_test/run_detection.py --output path/to/output

# 不保存可视化图片
python detection_test/run_detection.py --no-visualize

# 只保存 JSON 结果
python detection_test/run_detection.py --no-txt
```

## 输出结果

运行后，结果保存在 `detection_test/results/` 目录下：

```
detection_test/results/
├── result_000739.jpg          # 可视化结果图片
├── result_000901.jpg
├── ...
├── 000739.txt                 # YOLO 格式检测结果
├── 000901.txt
├── ...
└── detection_results.json     # 完整检测结果汇总
```

### JSON 结果格式

```json
{
  "model_path": "output/improved/stage3_refinement/weights/best.pt",
  "test_time": "2024-01-01T12:00:00",
  "total_images": 35,
  "total_detections": 127,
  "average_detections_per_image": 3.63,
  "conf_threshold": 0.25,
  "iou_threshold": 0.7,
  "results": [
    {
      "image": "000739.jpg",
      "detections": [
        {
          "class_id": 0,
          "class_name": "ship",
          "confidence": 0.8923,
          "bbox": [320.5, 240.3, 45.2, 30.1, 15.7]  // OBB: x_center, y_center, width, height, angle
        }
      ],
      "detection_count": 1
    }
  ]
}
```

### TXT 结果格式

每行一个检测结果：
```
class_id x_center y_center width height angle
```

示例：
```
0 320.500000 240.300000 45.200000 30.100000 15.700000
0 450.200000 300.400000 50.100000 35.200000 -5.300000
```

## 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `output/improved/stage3_refinement/weights/best.pt` | 模型路径 |
| `--images` | `detection_test/test_imgs` | 测试图片目录 |
| `--output` | `detection_test/results` | 输出目录 |
| `--conf` | `0.25` | 置信度阈值 (0-1) |
| `----iou` | `0.7` | IoU阈值 (0-1) |
| `--no-visualize` | False | 不保存可视化结果 |
| `--no-txt` | False | 不保存txt格式结果 |
| `--no-json` | False | 不保存json格式结果 |

## 注意事项

1. **模型格式**: 支持 `.pt` (PyTorch) 和 `.onnx` 格式
2. **OBB 支持**: 自动检测模型是否支持旋转框 (OBB)
3. **图片格式**: 支持 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
4. **置信度阈值**: 根据实际需求调整，默认 0.25
5. **IoU 阈值**: 用于 NMS，默认 0.7

