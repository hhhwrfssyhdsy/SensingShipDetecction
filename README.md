# YOLO11 海洋舰船检测改进项目

针对海洋舰船目标检测任务，对YOLO11进行算法改进，建立完整的训练和对比流程。

## 项目特点

- **简化设计**: 无复杂命令行参数，一键运行完整流程
- **三阶段渐进训练**: 60+40+30轮渐进优化
- **注意力机制**: SE -> CBAM -> MarineContext
- **自动对比**: 训练完成后自动评估并生成可视化图表

## 项目结构

```
SensingShipDetecction/
├── Config/
│   ├── config.py              # 训练配置
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
│   └── enhanced_neck.py       # 增强Neck
├── main.py                    # 主程序入口
├── trainer.py                 # 训练器
├── run_comparison.py          # 对比脚本
└── TECHNICAL_DOCUMENT.md      # 技术文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics matplotlib
```

### 2. 配置数据路径

编辑 `Config/config.py` 或设置环境变量:

```python
# 方式1: 直接修改 config.py
SSDD_TRAIN_INSHORE_IMG = Path("D:/DataSet/SSDD/train_inshore/images")
SSDD_TRAIN_LABEL = Path("D:/DataSet/SSDD/train_inshore/labels")

# 方式2: 环境变量
set SSDD_TRAIN_INSHORE_IMG=D:\DataSet\SSDD\train_inshore\images
set SSDD_TRAIN_LABEL=D:\DataSet\SSDD\train_inshore\labels
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
6. 模型对比评估
7. 生成可视化图表

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

## 输出结果

```
D:/ShipDetection_improved/
├── baseline/
│   └── baseline/
│       └── weights/
│           └── best.pt          # Baseline模型
├── improved/
│   ├── stage1_foundation/       # 第一阶段
│   ├── stage2_enhancement/      # 第二阶段
│   └── stage3_refinement/       # 第三阶段（最终模型）
│       └── weights/
│           └── best.pt
├── results/
│   ├── Baseline_YOLO11s_results.json
│   ├── Improved_MarineYOLO_results.json
│   └── comparison_report.json
├── visualization/
│   ├── comparison_metrics.png   # 指标对比柱状图
│   ├── comparison_radar.png     # 雷达图
│   └── improvement_percentage.png # 改进幅度图
└── training_history.json        # 训练历史
```

## 可视化图表

运行完成后自动生成:

1. **comparison_metrics.png** - 四指标对比柱状图
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
```

### 数据路径配置

支持的数据集:
- SSDD (SAR Ship Detection Dataset)
- SeaShips

自动统一为旋转框格式 (YOLO OBB)。

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

## 评估指标

- **mAP@0.5**: IoU=0.5时的平均精度
- **mAP@0.5:0.95**: COCO标准mAP
- **Precision**: 精确率
- **Recall**: 召回率

## 注意事项

1. **显存要求**: 建议8GB以上显存
2. **训练时间**: 完整训练约需数小时
3. **数据格式**: 自动处理水平框和旋转框
4. **公平对比**: Baseline和改进模型总轮数相同

## 技术细节

详见 [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)

## 许可证

MIT License
