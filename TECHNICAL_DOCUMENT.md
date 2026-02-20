# YOLO11 海洋舰船检测改进算法技术文档

## 1. 项目概述

本项目针对海洋舰船目标检测任务，对YOLO11进行算法改进，并建立完整的对比实验流程。

### 核心目标
- 训练标准YOLO11s作为Baseline
- 训练改进后的YOLO11（三阶段渐进训练 + 注意力机制）
- 自动对比两个模型的性能并生成可视化图表

---

## 2. 项目结构

```
SensingShipDetecction/
├── Config/
│   ├── config.py              # 训练配置
│   └── ship_detection.yaml    # 数据集配置
├── Data/
│   ├── dataset_analyzer.py    # 数据集分析
│   └── prepare_dataset.py     # 数据预处理
├── evaluation/                # 评估模块（可选）
├── models/                    # 模型改进模块
│   ├── attention_modules.py   # 注意力机制
│   ├── custom_yolo.py         # 自定义YOLO
│   └── enhanced_neck.py       # 改进Neck
├── test/                      # 测试模块
├── main.py                    # 主程序入口
├── trainer.py                 # 训练器
├── run_comparison.py          # 对比与可视化
└── README.md
```

---

## 3. 核心改进

### 3.1 三阶段渐进训练

| 阶段 | 名称 | 轮数 | 注意力机制 | 目标 |
|------|------|------|-----------|------|
| Stage 1 | 基础训练 | 60 | SE Attention | 建立基础检测能力 |
| Stage 2 | 增强训练 | 40 | CBAM | 提升小目标检测 |
| Stage 3 | 精细优化 | 30 | MarineContext | 海洋场景特化 |

**总轮数**: 130轮（与Baseline相同）

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

## 4. 使用方法

### 4.1 完整流程（推荐）

```bash
python main.py
```

自动执行：
1. 训练Baseline模型（130轮）
2. 训练改进模型（三阶段，共130轮）
3. 评估两个模型
4. 生成对比图表

### 4.2 单独训练

```bash
# 仅训练Baseline
python main.py --mode baseline

# 仅训练改进模型
python main.py --mode improved

# 仅对比（使用已训练模型）
python main.py --mode compare
```

### 4.3 单独对比

```bash
python run_comparison.py
```

---

## 5. 输出结果

### 5.1 模型文件

```
/root/ShipDetection_improved/
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
└── results/
    ├── baseline_results.json
    └── improved_progressive_results.json
```

### 5.2 可视化图表

```
/root/ShipDetection_improved/visualization/
├── comparison_metrics.png       # 指标对比柱状图
├── comparison_radar.png         # 雷达图
└── improvement_percentage.png   # 改进幅度图
```

### 5.3 对比报告

```
/root/ShipDetection_improved/results/
└── comparison_report.json       # 详细对比数据
```

---

## 6. 配置说明

### 6.1 环境变量

在运行前设置以下环境变量（可选，使用默认值）：

```bash
# 数据集路径
export SSDD_TRAIN_INSHORE_IMG="/path/to/train_inshore"
export SSDD_TRAIN_OFFSHORE_IMG="/path/to/train_offshore"
export SSDD_TEST_INSHORE_IMG="/path/to/test_inshore"
export SSDD_TEST_OFFSHORE_IMG="/path/to/test_offshore"
export SEASHIP_IMG="/path/to/seaships"

# 标签路径
export SSDD_TRAIN_LABEL="/path/to/train_labels"
export SSDD_TEST_INSHORE_LABEL="/path/to/test_inshore_labels"
export SSDD_TEST_OFFSHORE_LABEL="/path/to/test_offshore_labels"
export SEASHIP_LABEL="/path/to/seaship_labels"

# 训练参数
export TARGET_SIZE=640
export BATCH_SIZE=16
export DEVICE="auto"

# 输出路径
export OUTPUT_ROOT="/path/to/output"
```

### 6.2 训练配置

训练配置已内置在 `Config/config.py` 中：

- **Baseline**: 130轮标准训练
- **改进模型**: 三阶段渐进训练（60+40+30=130轮）

无需修改代码即可运行。

---

## 7. 评估指标

### 7.1 主要指标

| 指标 | 说明 |
|------|------|
| mAP@0.5 | IoU=0.5时的平均精度 |
| mAP@0.5:0.95 | IoU从0.5到0.95的平均精度 |
| Precision | 精确率 |
| Recall | 召回率 |
| Fitness | 综合适应度分数 |

### 7.2 对比维度

1. **绝对提升**: 改进模型 - Baseline
2. **相对提升**: (改进模型 - Baseline) / Baseline × 100%

---

## 8. 可视化说明

### 8.1 comparison_metrics.png
四张子图分别展示：
- mAP@0.5 对比
- mAP@0.5:0.95 对比
- Precision 对比
- Recall 对比

### 8.2 comparison_radar.png
雷达图展示四个维度的综合性能对比。

### 8.3 improvement_percentage.png
横向柱状图展示各项指标的改进幅度（百分比）。

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

### 9.3 数据增强演进

从保守到激进的数据增强策略：
- Stage 1: 基础增强，稳定学习
- Stage 2: 中等增强，提升泛化
- Stage 3: 强增强，精细优化

---

## 10. 注意事项

1. **显存要求**: 建议使用至少8GB显存的GPU
2. **训练时间**: 完整训练约需数小时（取决于硬件）
3. **数据准备**: 确保数据集路径正确配置
4. **模型路径**: 改进模型使用第三阶段的best.pt

---

## 11. 扩展建议

如需进一步改进：

1. **调整阶段配置**: 修改 `PROGRESSIVE_CONFIG`
2. **更换注意力机制**: 修改 `models/attention_modules.py`
3. **调整数据增强**: 修改各阶段的增强参数
4. **增加评估指标**: 扩展 `run_comparison.py`

---

## 12. 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置数据路径（可选）
export SSDD_TRAIN_INSHORE_IMG="/your/path"
# ... 其他路径

# 3. 运行完整流程
python main.py

# 4. 查看结果
ls /root/ShipDetection_improved/visualization/
```

---

**文档版本**: 1.0  
**更新日期**: 2026-02-20
