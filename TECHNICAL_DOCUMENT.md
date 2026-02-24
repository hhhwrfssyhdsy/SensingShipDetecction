# YOLO11 海洋舰船检测改进算法技术文档 - 边缘设备优化版

## 1. 项目概述

本项目针对海洋舰船目标检测任务，对YOLO11进行算法改进和边缘设备优化，建立完整的训练、优化和部署流程。

### 核心目标
- 训练标准YOLO11s作为Baseline
- 训练改进后的YOLO11（三阶段渐进训练 + 注意力机制）
- 训练边缘设备优化模型（轻量化 + 剪枝 + 量化）
- 自动对比三个模型的性能并生成可视化图表
- 导出ONNX模型用于边缘设备部署

---

## 2. 项目结构

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
├── models/                    # 模型改进模块
│   ├── attention_modules.py   # 注意力机制
│   ├── custom_yolo.py         # 自定义YOLO
│   ├── enhanced_neck.py       # 改进Neck
│   └── edge_optimization.py   # ⭐ 边缘优化模块（新增）
├── test/                      # 测试模块
│   ├── test_edge_standalone.py # 边缘优化测试
│   └── ...
├── main.py                    # 主程序入口（边缘优化版）
├── trainer.py                 # 训练器（含边缘训练）
├── run_comparison.py          # 对比与可视化
├── README.md                  # 使用说明
└── TECHNICAL_DOCUMENT.md      # 技术文档（本文档）
```

---

## 3. 核心改进

### 3.1 三阶段渐进训练

| 阶段 | 名称 | 轮数 | 注意力机制 | 目标 |
|------|------|------|-----------|------|
| Stage 1 | 基础训练 | 60 | SE Attention | 建立基础检测能力 |
| Stage 2 | 增强训练 | 40 | CBAM | 提升小目标检测 |
| Stage 3 | 精细优化 | 30 | MarineContext | 海洋场景特化 |

**总轮数**: 130轮（与Baseline相同，确保公平对比）

### 3.2 注意力机制

1. **SE Attention (Squeeze-and-Excitation)**
   - 通道注意力机制
   - 增强特征表达能力
   - 应用于Stage 1

2. **CBAM (Convolutional Block Attention Module)**
   - 通道 + 空间注意力
   - 提升小目标检测性能
   - 应用于Stage 2

3. **MarineContext Attention**
   - 针对海洋场景设计
   - 融合上下文信息
   - 应用于Stage 3

### 3.3 数据增强策略

各阶段采用渐进式数据增强：

| 阶段 | Mosaic | Mixup | Copy-Paste | Box Loss |
|------|--------|-------|-----------|----------|
| Stage 1 | 0.2 | 0.02 | - | 5.0 |
| Stage 2 | 0.5 | 0.1 | - | 6.5 |
| Stage 3 | 0.8 | 0.2 | 0.15 | 8.0 |

---

## 4. ⭐ 边缘设备优化（新增）

### 4.1 优化技术栈

| 技术 | 模块 | 说明 | 效果 |
|------|------|------|------|
| **轻量化模块** | `DepthwiseSeparableConv` | 深度可分离卷积 | 参数量减少50%+ |
| **轻量化模块** | `GhostModule` | Ghost模块 | 参数量减少40%+ |
| **模型剪枝** | `ModelPruner` | 结构化剪枝 | 模型大小减少30%+ |
| **INT8量化** | `ModelQuantizer` | 后训练量化 | 推理速度提升2-4x |
| **ONNX导出** | `EdgeOptimizer.export_to_onnx()` | 跨平台部署 | 支持多种推理框架 |

### 4.2 边缘优化配置

```python
# Config/config.py

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

# 边缘优化训练配置
EDGE_TRAINING_CONFIG = {
    "epochs": 100,                # 边缘模型训练轮数
    "batch_size": 4,              # 较小的批次大小
    "learning_rate": 0.001,       # 较低的学习率
    "weight_decay": 0.0005,
    "label_smoothing": 0.1,
    "dropout": 0.1,               # 添加dropout防止过拟合
}


```

### 4.3 边缘优化训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    边缘优化训练流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  步骤1: 训练基础模型 (100轮)                                  │
│  ├─ 使用改进模型架构 (SE Attention)                          │
│  ├─ 较小批次 (batch_size=4)                                  │
│  └─ 添加dropout防止过拟合                                    │
│                                                              │
│  步骤2: 应用边缘优化                                          │
│  ├─ 替换标准卷积为轻量化模块                                  │
│  │   ├─ DepthwiseSeparableConv                              │
│  │   └─ GhostModule                                         │
│  ├─ 模型剪枝 (移除20%不重要通道)                             │
│  └─ 量化准备 (INT8)                                          │
│                                                              │
│  步骤3: 导出ONNX模型                                          │
│  ├─ 生成edge_optimized.onnx                                 │
│  └─ 保存部署信息 (deployment_info.json)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 轻量化模块详解

#### 4.4.1 DepthwiseSeparableConv

深度可分离卷积将标准卷积分解为两步：

```
标准卷积: 输入 (C_in, H, W) -> 卷积核 (C_out, C_in, K, K) -> 输出 (C_out, H, W)
         参数量: C_out × C_in × K × K

深度可分离卷积:
  步骤1 - 深度卷积: 输入 (C_in, H, W) -> 分组卷积 (C_in, 1, K, K) -> 中间 (C_in, H, W)
  步骤2 - 点卷积: 中间 (C_in, H, W) -> 1x1卷积 (C_out, C_in, 1, 1) -> 输出 (C_out, H, W)
  参数量: C_in × K × K + C_out × C_in × 1 × 1

参数量减少比例: (K² + C_out) / (C_out × K²) ≈ 1/C_out (当K=3时)
```

**优势**:
- 参数量减少约50-60%
- 计算量(FLOPs)减少约50-60%
- 精度损失通常<1%

#### 4.4.2 GhostModule

Ghost模块通过廉价操作生成更多特征图：

```
输入特征: X (C_in, H, W)

步骤1: 生成固有特征图
  Y_primary = Conv(X)  ->  (C_out/ratio, H, W)

步骤2: 生成Ghost特征图（廉价操作）
  Y_ghost = CheapOperation(Y_primary)  ->  (C_out×(ratio-1)/ratio, H, W)

步骤3: 拼接
  Y = Concat(Y_primary, Y_ghost)  ->  (C_out, H, W)
```

**优势**:
- 参数量减少约40%
- 计算量减少约50%
- 精度损失通常<2%

### 4.5 模型剪枝详解

#### 4.5.1 结构化剪枝

```
原始卷积层: Conv(C_in, C_out, K, K)
           输出通道: [0, 1, 2, ..., C_out-1]

剪枝后: Conv(C_in, C_out×(1-ratio), K, K)
       移除重要性最低的 ratio×C_out 个通道

重要性计算: L1范数 ||W||_1 = Σ|w_ij|
           对每个输出通道的权重计算L1范数
           范数越小，重要性越低
```

#### 4.5.2 剪枝流程

1. **计算重要性**: 对每个卷积层的每个输出通道计算L1范数
2. **排序**: 按重要性分数排序
3. **选择**: 选择重要性最低的 `ratio×C_out` 个通道
4. **移除**: 从权重矩阵中移除对应通道
5. **微调**: 可选的微调训练恢复精度

### 4.6 INT8量化详解

#### 4.6.1 量化原理

```
浮点权重: W_float ∈ [-1.0, 1.0]
量化公式: W_int8 = round(W_float / scale + zero_point)

其中:
  scale = (max(W) - min(W)) / 255
  zero_point = -round(min(W) / scale)

反量化: W_dequant = (W_int8 - zero_point) × scale
```

#### 4.6.2 量化类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **PTQ** (Post-Training Quantization) | 训练后量化 | 快速部署，精度损失较小 |
| **QAT** (Quantization-Aware Training) | 量化感知训练 | 高精度要求 |

本项目使用PTQ，流程：
1. 准备量化配置
2. 在校准数据集上运行（收集统计信息）
3. 转换为量化模型

### 4.7 ONNX导出与部署

#### 4.7.1 ONNX导出

```python
# 导出流程
model = YOLO('path/to/best.pt')
torch.onnx.export(
    model.model,           # 模型
    dummy_input,           # 示例输入 (1, 3, 640, 640)
    'model.onnx',          # 输出路径
    opset_version=11,      # ONNX算子集版本
    input_names=['images'],
    output_names=['output0'],
    dynamic_axes={...}     # 动态轴（支持不同batch size）
)
```

#### 4.7.2 边缘设备部署

**Jetson Nano/Xavier**:
```bash
# 转换为TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine

# 推理
python -c "
import tensorrt as trt
import pycuda.driver as cuda
# TensorRT推理代码
"
```

**Raspberry Pi**:
```bash
# 使用ONNX Runtime
pip install onnxruntime

python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_image})
"
```

---

## 5. 使用方法

### 5.1 完整流程（推荐）

```bash
python main.py
```

自动执行：
1. 环境检查（含边缘优化配置）
2. 数据集分析
3. 数据准备
4. 训练Baseline模型（130轮）
5. 训练改进模型（三阶段，共130轮）
6. **训练边缘优化模型（100轮 + 优化）** ⭐
7. 三模型对比评估
8. 导出ONNX部署模型
9. 生成可视化图表

### 5.2 仅训练边缘优化模型

```python
from trainer import train_edge_only
train_edge_only()
```

### 5.3 单独对比

```bash
python run_comparison.py
```

---

## 6. 输出结果

### 6.1 模型文件

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
├── edge_optimized/                  # ⭐ 边缘优化模型
│   ├── base_training/
│   │   └── weights/
│   │       └── best.pt
│   └── edge_results.json
├── edge_deployment/                 # ⭐ 边缘部署文件
│   ├── edge_optimized.onnx          # ONNX模型
│   └── deployment_info.json         # 部署信息
├── results/
│   ├── Baseline_YOLO11s_results.json
│   ├── Improved_MarineYOLO_results.json
│   ├── EdgeOptimized_YOLO_results.json  # ⭐
│   └── comparison_report.json
├── visualization/
│   ├── comparison_metrics.png       # 三模型指标对比柱状图
│   ├── comparison_radar.png         # 雷达图
│   └── improvement_percentage.png   # 改进幅度图
└── training_history.json            # 训练历史
```

### 6.2 边缘部署信息 (deployment_info.json)

```json
{
    "model_info": {
        "name": "EdgeOptimized_YOLO",
        "description": "针对边缘设备优化的YOLO11模型",
        "optimization_techniques": [
            "轻量化模块 (DepthwiseSeparableConv, GhostModule)",
            "模型剪枝 (20%通道)",
            "INT8量化准备",
            "注意力机制 (SE -> CBAM -> MarineContext)"
        ]
    },
    "performance": {
        "accuracy": {
            "mAP50": 0.8745,
            "mAP50_95": 0.7123,
            "precision": 0.8712,
            "recall": 0.8456
        },
        "model_info": {
            "model_size_mb": 12.45,
            "num_parameters": 3456789
        }
    },
    "deployment": {
        "onnx_model": "edge_deployment/edge_optimized.onnx",
        "target_platforms": ["cpu", "gpu", "jetson", "raspberry_pi", "openvino"],
        "recommended_platform": "jetson"
    }
}
```

---

## 7. 配置说明

### 7.1 环境变量

```bash
# 训练参数
export TARGET_SIZE=640
export BATCH_SIZE=8
export DEVICE="0"

# 输出路径
export OUTPUT_ROOT="/path/to/output"

# ⭐ 边缘优化参数
export EDGE_OPTIMIZATION_ENABLED="true"
export EDGE_PRUNING_RATIO="0.2"
export EDGE_QUANTIZATION_BITS="8"
```

### 7.2 数据集准备

**数据集需人工准备**，放置在项目根目录下的 `dataset` 文件夹中：

```
dataset/
├── train/
│   ├── images/          # 训练图片 (.jpg/.png/.jpeg)
│   └── labels/          # 训练标签 (.txt)
├── val/
│   ├── images/          # 验证图片
│   └── labels/          # 验证标签
└── test/
    ├── images/          # 测试图片
    └── labels/          # 测试标签
```

**标签格式：** YOLO OBB格式
```
class x_center y_center width height angle
```
- 角度单位：**度（degrees）**
- 水平框角度设为 **0**

**推荐数据集:**
- **SSDD** (SAR Ship Detection Dataset) - 近岸/离岸SAR舰船检测数据集
- **RSDD-SAR** (Remote Sensing Ship Detection Dataset) - 遥感SAR舰船检测数据集

### 7.3 训练配置

```python
# Config/config.py

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

# ⭐ 边缘优化配置
EDGE_OPTIMIZATION = {
    "enabled": True,
    "use_lightweight_blocks": True,
    "use_pruning": True,
    "use_quantization": True,
    "pruning_ratio": 0.2,
    "quantization_bits": 8,
}
```

---

## 8. 评估指标

### 8.1 精度指标

| 指标 | 说明 |
|------|------|
| mAP@0.5 | IoU=0.5时的平均精度 |
| mAP@0.5:0.95 | IoU从0.5到0.95的平均精度 |
| Precision | 精确率 |
| Recall | 召回率 |

### 8.2 边缘设备性能指标 ⭐

| 指标 | 说明 |
|------|------|
| Model Size | 模型大小 (MB) |
| Parameters | 参数量 |

### 8.3 三模型对比维度

1. **Baseline vs Improved**: 算法改进效果
2. **Improved vs Edge**: 边缘优化代价
3. **Baseline vs Edge**: 端到端提升

---

## 9. 算法改进原理

### 9.1 渐进式训练优势

1. **稳定性**: 分阶段训练避免过拟合
2. **针对性**: 每阶段专注不同能力提升
3. **累积效应**: 后续阶段基于前期成果继续优化

### 9.2 注意力机制作用

1. **SE Attention**: 增强通道特征表达
2. **CBAM**: 同时关注通道和空间信息
3. **MarineContext**: 利用海洋场景上下文

### 9.3 边缘优化原理 ⭐

1. **轻量化**: 减少参数量和计算量
2. **剪枝**: 移除冗余通道，保持精度
3. **量化**: 降低精度位数，提升速度
4. **ONNX**: 标准化格式，跨平台部署

---

## 10. 性能对比示例

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

---

## 11. 注意事项

1. **显存要求**: 建议使用至少8GB显存的GPU
2. **训练时间**: 完整训练约需数小时（取决于硬件）
3. **数据准备**: 确保数据集已人工准备好，放置在 `dataset` 文件夹下
4. **边缘优化**: 优化后模型更适合嵌入式部署
5. **精度损失**: 边缘优化通常带来<3%的精度损失

---

## 12. 单元测试

### 12.1 运行所有测试

```bash
python test/run_all_tests.py
```

### 12.2 运行边缘优化测试

```bash
python test/test_edge_standalone.py
```

### 12.3 测试覆盖

| 测试文件 | 测试内容 |
|----------|----------|
| test_config.py | 配置验证 |
| test_data.py | 数据处理 |
| test_models.py | 模型模块 |
| test_edge_standalone.py | 边缘优化 |
| test_evaluation.py | 评估模块 |
| test_trainer.py | 训练器 |

---

## 13. 扩展建议

如需进一步改进：

1. **调整阶段配置**: 修改 `PROGRESSIVE_STAGES`
2. **更换注意力机制**: 修改 `models/attention_modules.py`
3. **调整边缘优化**: 修改 `EDGE_OPTIMIZATION` 配置
4. **更换轻量化模块**: 修改 `models/edge_optimization.py`
5. **调整剪枝比例**: 修改 `pruning_ratio` 参数
6. **尝试QAT量化**: 在 `ModelQuantizer` 中启用训练感知量化

---

## 14. 快速开始

```bash
# 1. 安装依赖
pip install ultralytics matplotlib

# 2. 准备数据集
# 将数据集放置在项目根目录的 dataset/ 文件夹下
# 结构：
#   dataset/
#   ├── train/images/ 和 train/labels/
#   ├── val/images/ 和 val/labels/
#   └── test/images/ 和 test/labels/

# 3. 运行完整流程
python main.py

# 4. 查看结果
ls <PROJECT_ROOT>/output/
```

---

## 15. 更新日志

### v2.0 - 边缘设备优化版 (2026-02-20)
- ✅ 新增边缘优化模块 (edge_optimization.py)
- ✅ 新增轻量化网络模块 (DepthwiseSeparableConv, GhostModule)
- ✅ 新增模型剪枝功能 (ModelPruner)
- ✅ 新增INT8量化支持 (ModelQuantizer)
- ✅ 新增ONNX导出功能
- ✅ 集成边缘优化到主流程 (main.py)

- ✅ 新增部署信息导出 (deployment_info.json)
- ✅ 完善单元测试 (test_edge_standalone.py)
- ✅ 更新技术文档 (本文档)

### v1.0 - 初始版本
- ✅ 三阶段渐进训练
- ✅ 注意力机制改进
- ✅ 自动对比评估
- ✅ 可视化图表生成

---

**文档版本**: 2.0  
**更新日期**: 2026-02-20  
**作者**: Ship Detection Team
