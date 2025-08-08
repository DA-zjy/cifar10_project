import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积模块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,      # 输入通道数 (灰度图为1)
                out_channels=16,    # 输出通道数
                kernel_size=3,      # 卷积核大小
                stride=1,           # 步长
                padding=1,          # 填充 (为了保持图片尺寸不变)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,      # 输入通道数 (灰度图为1)
                out_channels=32,    # 输出通道数
                kernel_size=3,      # 卷积核大小
                stride=1,           # 步长
                padding=1,          # 填充 (为了保持图片尺寸不变)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 池化层，图片尺寸减半
        )

        # 卷积模块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,      # 输入通道数 (灰度图为1)
                out_channels=128,    # 输出通道数
                kernel_size=3,      # 卷积核大小
                stride=1,           # 步长
                padding=1,          # 填充 (为了保持图片尺寸不变)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 卷积模块3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接分类器
        # 经过两次2*2*2的池化，32*32的图片变成了 4*4
        # 512是最后一个卷积层的输出通道数
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.6),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
        
        