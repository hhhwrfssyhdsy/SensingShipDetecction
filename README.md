# 船舶目标检测系统 (Ship Detection System)

基于YOLOv11-s的船舶目标检测系统，针对海洋环境中的小目标检测进行了专门优化。

## 项目结构

```
ShipDetection/
├── config.py              # 全局配置管理
├── dataset_analyzer.py    # 数据集分析工具
├── main.py               # 主程序入口
├── prepare_dataset.py    # 数据集准备工具
├── trainer.py            # 训练器
├── pyproject.toml        # 项目依赖
└── README.md            # 项目文档
```

## 核心特性

### 1. 智能训练策略选择
- **自动分析**: 基于数据集中小目标分布自动选择最优训练策略
- **多策略支持**: 标准训练、小目标优化、海洋环境优化、渐进式训练、协同优化

### 2. 渐进式训练框架
- **两阶段训练**: 基础训练 → 小目标强化
- **三阶段训练**: 正常目标 → 平衡训练 → 小目标强化
- **协同优化**: 数据增强 + 模型结构 + 注意力机制 三重优化

### 3. 海洋环境适配
- 针对海洋场景的光照变化、海浪模糊、传感器噪声等特性优化
- 小目标检测专用参数配置

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 uv
uv pip install -e .
```

### 数据集准备

项目支持SSDD和SeaShips数据集：

```bash
# 准备数据集
python main.py --prepare-only

# 或跳过准备直接训练（如果已准备好）
python main.py --skip-prepare
```

### 训练模型

```bash
# 自动选择训练策略
python main.py

# 使用特定策略
python main.py --strategy synergistic

# 仅分析数据集
python main.py --analyze-only
```

## 训练策略说明

| 策略 | 适用场景 | 特点 |
|------|---------|------|
| `standard` | 通用场景 | 标准YOLOv11-s训练参数 |
| `small_object` | 小目标较多 | 增强小目标检测能力 |
| `marine` | 海洋环境 | 针对海洋场景优化 |
| `two_stage` | 渐进优化 | 先基础后强化 |
| `three_stage` | 精细优化 | 分阶段逐步提升 |
| `synergistic` | 高难场景 | 数据+模型+注意力协同 |

## 配置说明

### 环境变量配置

可以通过环境变量覆盖默认路径：

```bash
# 数据集路径
export SSDD_TRAIN_INSHORE_IMG=/path/to/ssdd/train/inshore
export SSDD_TRAIN_OFFSHORE_IMG=/path/to/ssdd/train/offshore
export SEASHIP_IMG=/path/to/seaships

# 训练参数
export BATCH_SIZE=16
export EPOCHS=100
export DEVICE=0  # GPU设备号，或 'cpu'
```

### 配置文件

主要配置在 [config.py](config.py) 中：

- `TRAINING_STRATEGIES`: 各训练策略的参数配置
- `PROGRESSIVE_STAGES`: 渐进式训练的阶段配置
- `Config`: 路径和基础参数配置

## 算法原理

### 1. 数据集分析

系统自动分析数据集中的目标分布：

```python
# 目标尺寸分类
小目标: 相对面积 < 0.5% (32x32 @ 640x640)
中目标: 相对面积 0.5% - 2%
大目标: 相对面积 > 2%

# 海洋难度评分
marine_difficulty = 小目标比例 × 0.7 + 中目标比例 × 0.3
```

### 2. 训练策略选择逻辑

```
小目标比例 > 40%  →  协同优化训练 (synergistic)
小目标比例 > 25%  →  三阶段渐进训练 (three_stage)
小目标比例 > 15%  →  两阶段渐进训练 (two_stage)
其他               →  标准训练 (standard)
```

### 3. 渐进式训练流程

**三阶段训练示例**:

1. **阶段1 - 基础训练** (60 epochs)
   - 目标: 建立基础检测能力
   - 数据增强: 保守 (mosaic=0.2)
   - 学习率: 0.01

2. **阶段2 - 平衡训练** (40 epochs)
   - 目标: 平衡各类目标检测
   - 数据增强: 中等 (mosaic=0.5)
   - 学习率: 0.005

3. **阶段3 - 小目标强化** (30 epochs)
   - 目标: 专门提升小目标检测
   - 数据增强: 强 (mosaic=0.8)
   - 学习率: 0.001

### 4. 关键超参数

| 参数 | 标准值 | 小目标优化值 | 说明 |
|------|--------|-------------|------|
| mosaic | 0.3 | 0.7-0.8 | Mosaic增强概率 |
| mixup | 0.05 | 0.15-0.2 | MixUp增强概率 |
| box | 5.0 | 7.5-8.0 | 边界框损失权重 |
| cls | 1.0 | 0.5-0.6 | 分类损失权重 |
| dfl | 1.5 | 2.0 | 分布焦点损失权重 |
| anchor_t | 3.0 | 3.0 | 锚点阈值 |
| fl_gamma | 0.0 | 1.5 | 焦点损失gamma |

## API使用

### 作为模块使用

```python
from config import Config
from dataset_analyzer import DatasetAnalyzer
from trainer import ShipDetectionTrainer
from prepare_dataset import prepare_ship_dataset

# 准备数据集
prepare_ship_dataset()

# 分析数据集
analyzer = DatasetAnalyzer(img_paths, label_paths)
stats = analyzer.analyze()
strategy = analyzer.recommend_strategy()

# 训练
trainer = ShipDetectionTrainer()
model, results = trainer.train(strategy="synergistic")
```

## 依赖项

- Python >= 3.11
- PyTorch >= 2.8.0
- ultralytics >= 8.3.223
- albumentations >= 2.0.8

完整依赖见 [pyproject.toml](pyproject.toml)

## 许可证

MIT License
