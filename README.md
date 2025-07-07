# vit-attention-cifar
A PyTorch project combining ResNet, attention mechanisms and MobileViT for image classification on CIFAR-10
# 🔍 基于注意力机制与 Transformer 的图像分类模型设计与对比实验

本项目旨在对比不同类型的注意力机制（SE、CBAM、ECA）与融合 Transformer 模块（MobileViT）对图像分类任务性能的影响。我们基于 ResNet18/50 架构，在 CIFAR-10 数据集上进行了多组模型结构实验。

---

## 🧱 实验结构设计

本项目支持以下模型结构：

| 模型名称          | 描述说明                                 |
| ----------------- | ---------------------------------------- |
| `resnet18`        | 基础卷积残差网络（浅层）                 |
| `resnet50`        | 深层残差网络                             |
| `resnet18 + SE`   | 加入通道注意力（Squeeze-and-Excitation） |
| `resnet18 + CBAM` | 通道+空间注意力（CBAM）                  |
| `resnet18 + ECA`  | 高效通道注意力模块（ECA）                |
| `mobilevit`       | 卷积 + Transformer 融合模块              |

所有模型均通过 YAML 文件配置，训练轮数统一设置为 300。

---

## 🧪 数据集与环境

- 数据集：CIFAR-10（自动下载）
- 图像大小：3×32×32
- 分类数：10 类
- 开发环境：
  - Python 3.8+
  - PyTorch >= 1.10
  - CUDA 支持（可选）
  - `wandb`（可选，用于记录训练过程）

---

## 🚀 运行方法

### ✅ 单模型训练

```bash
python main.py --config config/baseline/resnet18.yaml
```
---
### ✅ 批量实验运行
```bash
chmod +x run.sh
./run.sh
```
---
### ✅ 配置文件样例（config/baseline/resnet18_cbam.yaml）
```yaml
model: resnet18
dataset: CIFAR10
num_classes: 10
lr: 0.01
optimizer: Adam
batch_size: 128
epochs: 200
base_channels: 64
attention: CBAM
```
---

### 项目结构
```text

├── main.py                   # 主训练入口
├── models/
│   ├── resnet.py             # ResNet18/50 + attention支持
│   ├── resnet_mobilevit.py   # MobileViT模块定义
│   └── attention.py          # SELayer / CBAM / ECA 实现
├── config/
│   └── baseline/*.yaml       # 各模型对应配置文件
├── run.sh                    # 批量运行脚本
├── utils/
│   └── progress_bar.py       # 简易训练进度条
└── logs/                     # 每次运行日志输出
```
---

### 📊 实验指标记录建议
模型结构	参数量(M)	Top-1 准确率	最优 epoch
ResNet18	11M	84.95%	-
ResNet18 + SE	11.3M	90.3%	-
ResNet18 + CBAM	11.5M	92.62%	-
ResNet18 + ECA	11.2M	90.74%	-
ResNet50	23M	90.04%	-
MobileViTResNet	~13M	93.33%	-

---

### 📚 参考文献
[1] He K, et al., "Deep Residual Learning for Image Recognition", CVPR 2016.

[2] Hu J, et al., "Squeeze-and-Excitation Networks", CVPR 2018.

[3] Woo S, et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

[4] Wang Q, et al., "ECA-Net", CVPR 2020.

[5] Mehta S, Rastegari M, "MobileViT", ICLR 2022.
