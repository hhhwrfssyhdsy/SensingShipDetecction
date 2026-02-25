# YOLO11 海洋舰船检测

针对海洋舰船目标检测任务，对YOLO11进行算法改进和边缘设备优化，建立完整的训练、优化和部署流程。

## 项目特点

- **简化设计**: 无复杂命令行参数，一键运行完整流程
- **三阶段渐进训练**: 60+40+30轮渐进优化
- **注意力机制**: SE -> CBAM -> MarineContext
- **边缘设备优化**: 轻量化、剪枝、量化、ONNX导出
- **自动对比**: 训练完成后自动评估并生成可视化图表

## 项目结构

```
SensingShipDetecction/
├── Config/
│   ├── config.py              # 训练配置（含边缘优化配置）
│   └── ship_detection.yaml    # 数据集配置
├── Data/
│   ├── dataset_analyzer.py    # 数据集分析
│   └── prepare_dataset.py     # 数据预处理
├── evaluation/                # 评估模块
│   ├── evaluator.py           # 模型评估
│   ├── metrics.py             # 指标计算
│   └── visualizer.py          # 可视化
├── models/                    # 模型改进
│   ├── attention_modules.py   # 注意力机制
│   ├── custom_yolo.py         # 自定义YOLO
│   ├── enhanced_neck.py       # 增强Neck
│   └── edge_optimization.py   # 边缘优化模块 
├── test/                      # 单元测试
│   ├── test_edge_standalone.py # 边缘优化测试
│   └── ...
├── main.py                    # 主程序入口
├── trainer.py                 # 训练器
├── run_comparison.py          # 对比脚本
└── TECHNICAL_DOCUMENT.md      # 技术文档
```

## 快速开始

### 1. 安装依赖

使用pip:
```bash 
pip install -r requirements.txt
```
使用uv:
```bash
uv sync
```


### 2. 配置数据路径

在项目根目录下，将数据集放在 `dataset` 文件夹下。
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 3. 运行完整流程

```bash
python main.py
```

自动执行:
1. 环境检查
2. 数据集分析
3. 数据准备
4. Baseline训练 (130轮)
5. 改进模型训练 (三阶段: 60+40+30轮)
6. 边缘设备优化
7. 模型对比评估（Baseline vs Improved）
8. 导出ONNX部署模型
9. 生成可视化图表

### 4. 仅运行对比（已有模型）

```bash
python run_comparison.py
```


## 三阶段渐进训练

| 阶段 | 轮数 | 注意力机制 | 数据增强 | 目标 |
|------|------|-----------|---------|------|
| Stage 1 | 60 | SE Attention | Mosaic 0.2, Mixup 0.02 | 基础检测能力 |
| Stage 2 | 40 | CBAM | Mosaic 0.5, Mixup 0.1 | 小目标检测 |
| Stage 3 | 30 | MarineContext | Mosaic 0.8, Mixup 0.2, Copy-Paste 0.15 | 海洋场景特化 |

**总轮数**: 130轮（与Baseline相同，公平对比）

## 边缘设备优化

### 优化技术

| 技术 | 说明 | 效果 |
|------|------|------|
| **轻量化模块** | DepthwiseSeparableConv, GhostModule | 参数量减少50%+ |
| **模型剪枝** | 移除不重要通道 | 模型大小减少30%+ |
| **INT8量化** | 降低精度提升速度 | 推理速度提升2-4x |
| **ONNX导出** | 跨平台部署 | 支持多种推理框架 |

### 边缘优化配置 (Config/config.py)

```python
EDGE_OPTIMIZATION = {
    "enabled": True,              # 启用边缘优化
    "use_lightweight_blocks": True,  # 使用轻量化模块
    "use_pruning": True,          # 启用模型剪枝
    "use_quantization": True,     # 启用INT8量化
    "pruning_ratio": 0.2,         # 剪枝比例 (0-1)
    "quantization_bits": 8,       # 量化位数
}

# 边缘设备目标平台
EDGE_TARGET_PLATFORMS = ["cpu", "gpu", "jetson", "raspberry_pi", "openvino"]
```

### 边缘优化训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    边缘优化训练流程                          │
├─────────────────────────────────────────────────────────────┤
│  1. 训练基础模型 (100轮)                                      │
│     └─> 使用改进模型 (SE Attention)                          │
│                                                              │
│  2. 应用边缘优化                                              │
│     ├─> 替换为轻量化模块                                      │
│     ├─> 模型剪枝 (20%通道)                                   │
│     └─> 量化准备                                             │
│                                                              │
│  3. 导出ONNX模型                                              │
│     └─> 边缘设备部署                                         │
└─────────────────────────────────────────────────────────────┘
```

## 输出结果

输出目录由 `Config.OUTPUT_ROOT` 决定，默认为项目目录下的 `output/` 文件夹：

```
<PROJECT_ROOT>/output/
├── baseline/
│   └── baseline/
│       └── weights/
│           └── best.pt              # Baseline模型
├── improved/
│   ├── stage1_foundation/           # 第一阶段
│   ├── stage2_enhancement/          # 第二阶段
│   └── stage3_refinement/           # 第三阶段（最终模型）
│       └── weights/
│           └── best.pt
├── edge_optimized/                  # 边缘优化模型
│   ├── base_training/
│   │   └── weights/
│   │       └── best.pt
│   └── edge_results.json
├── edge_deployment/                 # 边缘部署文件
│   ├── edge_optimized.onnx          # ONNX模型
│   └── deployment_info.json         # 部署信息
├── results/
│   ├── Baseline_YOLO11s_results.json
│   ├── Improved_MarineYOLO_results.json
│   ├── EdgeOptimized_YOLO_results.json  
│   └── comparison_report.json
├── visualization/
│   ├── comparison_metrics.png       # 指标对比柱状图
│   ├── comparison_radar.png         # 雷达图
│   └── improvement_percentage.png   # 改进幅度图
└── training_history.json            # 训练历史
```

## 可视化图表

运行完成后自动生成:

1. **comparison_metrics.png** - 三模型指标对比柱状图 (Baseline vs Improved vs Edge)
2. **comparison_radar.png** - 性能雷达图
3. **improvement_percentage.png** - 改进幅度百分比

## 配置说明

### 训练参数 (Config/config.py)

```python
# 基础配置
TARGET_SIZE = 640          # 输入尺寸
BATCH_SIZE = 8             # 批次大小
DEVICE = "0"               # GPU设备

# Baseline训练
BASELINE_EPOCHS = 130

# 渐进训练配置
PROGRESSIVE_STAGES = [
    {"epochs": 60, "attention": "se", ...},
    {"epochs": 40, "attention": "cbam", ...},
    {"epochs": 30, "attention": "marine", ...},
]

# 边缘优化配置
EDGE_OPTIMIZATION = {
    "enabled": True,
    "use_lightweight_blocks": True,
    "use_pruning": True,
    "use_quantization": True,
    "pruning_ratio": 0.2,
    "quantization_bits": 8,
}

EDGE_TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 0.001,
    "dropout": 0.1,
}
```

### 数据集说明
**标签格式：** YOLO OBB格式
```
class x_center y_center width height angle
```
- 角度单位：**度（rad）**
- 水平框角度设为 **0**

**推荐数据集:**
- **SSDD** (SAR Ship Detection Dataset) - 近岸/离岸SAR舰船检测数据集
- **RSDD-SAR** (Remote Sensing Ship Detection Dataset) - 遥感SAR舰船检测数据集
[整合数据集](https://www.kaggle.com/datasets/kirakiramika/ship-detection-dataset)


## 注意力机制

### 1. SE Attention (Squeeze-and-Excitation)
- 通道注意力机制
- 增强特征表达能力

### 2. CBAM (Convolutional Block Attention Module)
- 通道 + 空间注意力
- 提升小目标检测性能

### 3. MarineContext Attention
- 针对海洋场景设计
- 融合上下文信息

## 边缘优化模块详解

### DepthwiseSeparableConv

深度可分离卷积，将标准卷积分解为深度卷积和点卷积:
- **参数量**: 减少约50%
- **计算量**: 减少约60%
- **精度损失**: <1%

### GhostModule

Ghost模块，通过廉价操作生成更多特征图:
- **参数量**: 减少约40%
- **计算量**: 减少约50%
- **精度损失**: <2%

### ModelPruner

结构化剪枝，移除不重要的通道:
- **剪枝比例**: 可配置 (默认20%)
- **模型大小**: 减少约30%
- **推理速度**: 提升约20%

### ModelQuantizer

INT8量化，降低模型精度提升速度:
- **量化位数**: 8bit
- **模型大小**: 减少约75%
- **推理速度**: 提升2-4x
- **精度损失**: 通常<3%

## 评估指标

### 精度指标
- **mAP@0.5**: IoU=0.5时的平均精度
- **mAP@0.5:0.95**: COCO标准mAP
- **Precision**: 精确率
- **Recall**: 召回率

### 边缘设备性能指标
- **Model Size**: 模型大小 (MB)
- **Parameters**: 参数量

## 性能对比示例

```
📊 最终对比结果
============================================================

精度指标对比:
指标                 Baseline     Improved     Edge         提升
--------------------------------------------------------------------------------
mAP50                0.8234       0.8912       0.8745       +0.0511
mAP50_95             0.6543       0.7234       0.7123       +0.0580
precision            0.8123       0.8823       0.8712       +0.0589
recall               0.7890       0.8567       0.8456       +0.0566

边缘设备性能:
   模型大小: 12.45 MB
   参数量: 3,456,789
```



## 注意事项

1. **显存要求**: 建议8GB以上显存
2. **训练时间**: 完整训练约需数小时
3. **数据格式**: 自动处理水平框和旋转框
4. **公平对比**: Baseline、Improved、Edge总轮数相同
5. **边缘设备**: 优化后模型更适合嵌入式部署

## 部署到边缘设备

### Jetson Nano/Xavier

```bash
# 1. 导出TensorRT模型
python -c "
from models.edge_optimization import EdgeOptimizer
from ultralytics import YOLO

model = YOLO('path/to/edge_optimized.pt')
optimizer = EdgeOptimizer(model.model)
optimizer.export_tensorrt('model.engine')
"

# 2. 在Jetson上运行
# 使用 TensorRT 进行推理
```

### Raspberry Pi

```bash
# 使用ONNX Runtime进行推理
pip install onnxruntime

python -c "
import onnxruntime as ort
session = ort.InferenceSession('edge_optimized.onnx')
# 进行推理
"
```

## 许可证

MIT License
