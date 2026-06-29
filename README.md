# AI Image Detector — AI 生成图像检测系统

基于双分支神经网络（空间域 + 频域）的 AI 生成图像检测系统，用于区分真实图像与 AI 生成图像。

## 项目结构

```
ai-image-detector/
├── configs/
│   └── config.yaml              ← 训练参数配置文件
├── data/                         ← 数据集目录（需自行放入图片）
│   ├── real/                     ← 真实图片文件夹
│   └── fake/                     ← AI 生成图片文件夹
├── models/
│   ├── spatial_branch.py         ← 空间分支（ResNet34 多尺度）
│   ├── frequency_branch.py       ← 频域分支（FFT + 轻量 CNN）
│   ├── fusion.py                 ← 特征级门控融合
│   └── classifier.py             ← 分类头
├── utils/
│   ├── metrics.py                ← 评估指标
│   ├── logger.py                 ← 日志工具
│   └── fft_utils.py              ← FFT 工具函数
├── checkpoints/                  ← 训练模型保存目录
│   ├── best.pth                  ← 最佳模型（Acc=96.96%）
│   ├── checkpoint_last.pth       ← 最新状态（支持断点续训）
│   └── epoch_*.pth               ← 各轮权重
├── model.py                      ← 模型入口
├── train.py                      ← 训练脚本
├── eval.py                       ← 评估脚本
├── infer.py                      ← 单图推理脚本
├── requirements.txt              ← Python 依赖
└── README.md                     ← 本文件
```

## 环境配置

### 方式一：使用 Conda（推荐）

```bash
# 创建并激活环境
conda create -n pytorch python=3.10
conda activate pytorch

# 安装 PyTorch（含 CUDA 支持）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install pyyaml pillow
```

### 方式二：使用 pip

```bash
pip install -r requirements.txt
```

## 数据集准备

### 数据集目录结构

将图片按以下目录结构存放：

```
data/
├── real/          ← 真实图片（.jpg / .png）
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── fake/          ← AI 生成图片（.jpg / .png）
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

### 推荐数据集

- **CIFAKE**（Kaggle）：50,000 张真实图 + 50,000 张 AI 图，32×32 像素
  - 下载后解压，将 `REAL/` 和 `FAKE/` 文件夹放到 `data/` 目录下

## 训练模型

### 从头开始训练

```bash
conda activate pytorch
python train.py
```

训练过程会自动：
- 从 `data/real/` 和 `data/fake/` 加载图片
- 按 8:2 自动分割训练集和验证集
- 每轮结束后在验证集上评估并保存最佳模型
- 满足早停条件时自动停止

### 断点续训

如果训练中途中断，可以从中断处恢复：

```bash
python train.py --resume
```

### 修改训练参数

训练参数在 `configs/config.yaml` 中配置，常用参数：

```yaml
data:
  batch_size: 32        # 批处理大小
  num_workers: 2        # 数据加载线程数
  train_ratio: 0.8      # 训练集比例（剩余为验证集）

training:
  epochs: 20            # 最大训练轮数
  lr: 0.0003            # 初始学习率
  early_stop_patience: 5  # 早停耐心值
```

### 查看训练进度

```bash
# 查看最新输出
Get-Content train_error2.log -Tail 5

# 查看所有 epoch 的结果
Select-String "Train loss|done|Best" train_error2.log
```

## 评估模型

用验证集评估最佳模型的性能：

```bash
python eval.py
```

输出示例：

```
Device: cuda
Val samples: 20000
Loaded: checkpoints\best.pth

Evaluation Results:
  Loss:      0.0851
  Accuracy:  96.96%
  Precision: 96.99%
  Recall:    96.99%
  F1 Score:  96.99%
```

## 推理测试

### 测试单张图片

```bash
# 基本用法
python infer.py 图片路径.jpg --ckpt checkpoints/best.pth

# 示例：测试 data/ 下的图片，0.jpg为AI生成照片；0001.jpg为真实照片；
python infer.py data/xxx.jpg --ckpt checkpoints/best.pth

# 示例：测试桌面上的图片
python infer.py C:\Users\用户名\Desktop\照片.jpg --ckpt checkpoints/best.pth
```

输出示例：

```
Loaded checkpoint: checkpoints/best.pth
Result: AI-generated (confidence: 0.9987)
```

结果说明：
- `Real`：模型判定为真实图像
- `AI-generated`：模型判定为 AI 生成的图像
- `confidence`：置信度（0~1，越高越确定）

### 可选参数

```bash
python infer.py 图片路径.jpg --ckpt checkpoints/best.pth --device cpu
```

- `--ckpt`：指定模型权重文件路径（默认 `checkpoints/best.pth`）
- `--device`：推理设备，可选 `auto` / `cuda` / `cpu`（默认 `auto`）

## 项目特点

### 网络架构

- **双分支设计**：空间分支（ResNet34 多尺度）+ 频域分支（FFT + 轻量 CNN），捕获空间和频域两个维度的伪造特征
- **特征级门控融合**：为每个特征维度生成独立融合权重，比简单拼接更精细
- **梯度阻断**：FFT 使用 `torch.no_grad()` 阻断反向传播，加速训练

### 工程优化

- **AMP 混合精度**：训练速度提升约 2 倍
- **断点续训**：`--resume` 参数支持从断点恢复
- **早停机制**：连续 5 轮无提升自动停止
- **学习率调度**：ReduceLROnPlateau 自动调低学习率

### 实验结果

在 CIFAKE 数据集上的最佳结果（Epoch 9）：

| 指标 | 值 |
|------|-----|
| 准确率 | 96.96% |
| 精确率 | 96.99% |
| 召回率 | 96.99% |
| F1 分数 | 96.99% |

## 注意事项

1. 模型在 **CIFAKE 数据集**（32×32 小图）上训练，对高清真实照片的泛化能力有限，如需更好的泛化性能，建议使用高清数据集重新训练
2. 推理时请确保已激活正确的 Python 环境（`conda activate pytorch`）
3. 文件名包含括号等特殊字符时，请用引号包裹（如 `python infer.py "data/照片 (1).jpg"`）

## 依赖列表

```
torch >= 1.10.0
torchvision >= 0.11.0
pillow >= 9.0.0
pyyaml >= 5.0
```
