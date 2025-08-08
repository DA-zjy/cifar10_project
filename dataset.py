# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DATASET_PATH

def get_dataloaders():
    """返回训练和测试的数据加载器"""
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(10),     # 随机旋转-10到10度
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# dataset.py 的最下方

# 这是一个单元测试模块，只有在直接运行 `python dataset.py` 时才会执行
if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders()
    
    print("开始测试数据加载器...")
    
    # 从训练数据加载器中取出一个批次的数据
    train_features, train_labels = next(iter(train_loader))
    
    print(f"训练数据批次的特征形状: {train_features.shape}")
    print(f"训练数据批次的标签形状: {train_labels.shape}")
    
    # 打印其中一个样本的形状和标签
    img = train_features[0]
    label = train_labels[0]
    print(f"单个图片的形状: {img.shape}")
    print(f"单个图片的标签: {label}")
    
    print("-" * 20)
    
    # 从测试数据加载器中取出一个批次的数据
    test_features, test_labels = next(iter(test_loader))
    print(f"测试数据批次的特征形状: {test_features.shape}")
    print(f"测试数据批次的标签形状: {test_labels.shape}")
    
    print("数据加载器测试成功！")