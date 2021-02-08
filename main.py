import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from config import DEVICE
from model.resnet20 import ResNet20, BasicBlock
from transforms.transforms import transform
from utils.train_eval import train

BATCH_SIZE = 100

train_dataset = torchvision.datasets.CIFAR10(
    './cifar_10/', transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(
    './cifar_10/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ResNet20(BasicBlock).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

history = train(model, criterion, train_loader, test_loader,
                optimizer, scheduler, epochs=200)

torch.save(model, './saved_models/resnet20.pth')
