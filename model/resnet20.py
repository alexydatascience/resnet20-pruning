import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self.conv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                self.conv(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def conv(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        if self.shortcut:
            residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet20(nn.Module):
    def __init__(self, block, n_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, out_channels=16, stride=1)
        self.layer2 = self.make_layer(block, out_channels=32, stride=2)
        self.layer3 = self.make_layer(block, out_channels=64, stride=2)
        self.fc = nn.Linear(64, n_classes)

    def make_layer(self, block, out_channels, stride):
        layers = []
        layer = block(self.in_channels, out_channels, stride)
        layers.append(layer)
        self.in_channels = out_channels
        for i in range(2):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
