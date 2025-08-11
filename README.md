# PyTorch CIFAR-10 图像分类项目实践

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. 项目概述

本项目是一个端到端的深度学习实践项目，旨在利用 PyTorch 从零开始搭建、训练并系统性地优化一个卷积神经网络（CNN），以解决经典的 CIFAR-10 图像分类任务。

项目的核心不仅仅是达成一个高性能的分类器，更在于记录和展示一个从**基础模型**到**SOTA（State-of-the-Art）架构**，从**简单训练**到**高级优化策略**的完整学习和迭代过程。整个项目遵循了现代软件工程的最佳实践，采用模块化的代码结构和 Git 进行版本控制。

**最终，通过实现 ResNet-34 架构并结合多种高级训练技巧，模型在 CIFAR-10 测试集上达到了 92.21% 的准确率。**

## 2. 核心技术与特性

本项目实践并验证了深度学习中的一系列核心技术：

* **模型架构演进**:
    * `SimpleCNN`: 实现了一个基础的卷积神经网络。
    * `VGG-style`: 掌握了通过堆叠小的（3x3）卷积核来构建更深、更有效网络的设计思想。
    * `ResNet-34`: 从零开始完整实现了包含残差连接（Residual Connection）的 ResNet-34 架构，解决了深度网络训练中的退化问题。

* **高级训练策略**:
    * **优化器 (Optimizer)**: 对比实践了 `AdamW` 和 `SGD with Momentum` 的性能差异。
    * **学习率调度器 (Scheduler)**: 应用了 `CosineAnnealingLR` 实现学习率的平滑衰减，以寻找更优的最小值点。
    * **正则化 (Regularization)**:
        * `BatchNorm`: 加速模型收敛并提升稳定性。
        * `Dropout`: 防止模型在全连接层过拟合。
        * `Label Smoothing`: 防止模型过分自信，提升泛化能力。
        * `Weight Decay`: 对模型权重进行约束。

* **数据处理**:
    * **数据增强 (Data Augmentation)**: 采用了 `RandomCrop`, `RandomHorizontalFlip`, `AutoAugment`, `RandomErasing` 等一系列强大的增强策略来扩充数据集，极大地提升了模型的鲁棒性和泛化能力。
    * **测试时增强 (Test Time Augmentation - TTA)**: 在评估阶段通过对测试图片及其翻转版本进行集成预测，进一步提升了最终的准确率。

* **工程实践**:
    * **模块化设计**: 将项目代码解耦为 `config.py`, `dataset.py`, `model.py`, `train.py` 等多个模块，提高了代码的可读性和可维护性。
    * **版本控制**: 使用 **Git** 和 **GitHub** 对整个实验过程进行版本管理，通过**分支 (Branch)** 进行新功能开发，通过**标签 (Tag)** 记录关键的性能里程碑。

## 3. 性能与里程碑

本项目的亮点在于清晰的迭代优化路径，每一次架构或策略的升级都带来了显著的性能提升。

| 版本标签 | 核心架构 | 关键优化技术 | 峰值准确率 |
| :--- | :--- | :--- | :--- |
| `v1.0-simple-cnn` | 基础 CNN (5x5卷积核) | AdamW, 基础数据增强 | ~81% |
| `v2.0-vgg-style` | VGG-style CNN (3x3卷积核) | + 学习率调度器 | 88.51% |
| `v4.0-resnet` | **ResNet-34** (从零实现) | + SGD, 标签平滑, 高级数据增强 | 91.91% |
| `v5.0-resnet-tta` | **ResNet-34** | **+ 测试时增强 (TTA)** | **92.21%** |

*注：v3.0 为 ResNet 的初版，v4.0 为其优化版，v5.0 为最终版。*

## 4. 项目结构

```
cifar10_project/
├── checkpoints/           # 用于存放训练好的模型权重 (被.gitignore忽略)
├── data/                  # 用于存放 CIFAR-10 数据集 (被.gitignore忽略)
├── .gitignore             # Git 忽略规则文件
├── config.py              # 存放所有配置信息和超参数
├── dataset.py             # 负责数据的加载和预处理
├── model.py               # 定义 CNN / ResNet 模型结构
├── predict.py             # (可选) 加载已训练模型进行预测的脚本
├── README.md              # 项目说明文档
└── train.py               # 训练和评估模型的主脚本
```

## 5. 安装与环境设置

建议使用 Conda 来管理环境。

1.  **克隆仓库**
    ```bash
    git clone [https://github.com/DA-zjy/cifar10_project.git](https://github.com/DA-zjy/cifar10_project.git)
    cd cifar10_project
    ```

2.  **创建并激活 Conda 环境**
    ```bash
    # 我们创建一个新的、干净的环境，名叫 cifar10_env
    conda create -n cifar10_env python=3.9 -y
    conda activate cifar10_env
    ```

3.  **安装依赖**

    * **方法一（推荐）：使用 `requirements.txt` 一键安装**
      ```bash
      pip install -r requirements.txt
      ```
      **注意**: `requirements.txt` 中包含了 PyTorch 的 CPU 版本或你本地的 CUDA 版本。如果你的设备（比如没有 NVIDIA 显卡）需要不同版本的 PyTorch，请参考方法二。

    * **方法二：手动安装核心依赖**
      如果方法一出现问题（尤其是 PyTorch 的 CUDA 版本不匹配时），请手动安装：
      1.  **安装 PyTorch**: 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，根据你的系统和 CUDA 版本，获取并运行最适合你的安装命令。
      2.  **安装其他库**:
          ```bash
          pip install tqdm Pillow
          ```
          
## 6. 使用方法

### 训练模型
直接从命令行运行 `train.py` 脚本即可开始训练。
```bash
python train.py
```
* 所有超参数（如 Epochs, Batch Size, Learning Rate）都可以在 `config.py` 文件中进行修改。
* 训练过程中，性能最佳的模型权重将被自动保存到 `checkpoints/` 目录下，名为 `best_model.pth`。

### 进行预测
(如果已创建 `predict.py`)
1.  确保 `checkpoints/` 目录下已有训练好的模型文件（如 `best_model.pth`）。
2.  准备一张你想要测试的图片。
3.  修改 `predict.py` 脚本中的 `MODEL_TO_LOAD` 和 `IMAGE_TO_TEST` 路径。
4.  运行脚本：
    ```bash
    python predict.py
    ```

## 7. 未来可探索的方向
* **迁移学习**: 加载 `torchvision.models` 中在 ImageNet 上预训练的 ResNet 模型进行微调，与从零训练的结果进行对比。
* **探索更先进的架构**: 尝试实现 Vision Transformer (ViT) 或 ConvNeXt 等更现代的模型架构。
* **部署模型**: 使用 Flask 或 FastAPI 将训练好的模型包装成一个简单的 Web API 服务。