your_project/
├── main.py                         # 主训练入口（原始 + 修改版，支持 config）
├── config/
│   ├── baseline/
│   │   ├── cifar10.yaml           # 原始 ResNet 配置
│   │   ├── mobilevit.yaml         # MobileViT 配置（新增 ✅）
│   │   └── cbam.yaml              # ResNet + CBAM 配置（可选）
├── models/
│   ├── resnet.py                  # ResNet + Attention模块（SE/CBAM/ECA）
│   ├── attention.py              # SELayer, CBAM, ECALayer（你已写）
│   ├── resnet_mobilevit.py       # ✅ MobileViTResNet + MobileViTBlock（我写的）
│   └── __init__.py               # 可选（用于 import）
├── utils/
│   └── progress_bar.py           # 训练进度条打印函数（你已有）
├── data/
│   └── (下载的 CIFAR 数据集自动生成在这里)
├── checkpoint/
│   └── ckpt.pth                  # 自动保存模型 checkpoint
├── wandb/                        # wandb 跟踪文件夹（自动生成）
└── README.md                     # 项目说明文档（建议添加）
