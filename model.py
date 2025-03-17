import numpy as np 
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import tqdm


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)

        self.model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(32)

        self.model.layer1[0].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[0].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn2 = nn.BatchNorm2d(32)

        self.model.layer1[1].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[1].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn2 = nn.BatchNorm2d(32)

        self.model.layer2[0].conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer2[0].bn1 = nn.BatchNorm2d(64)
        self.model.layer2[0].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[0].bn2 = nn.BatchNorm2d(64)
        self.model.layer2[0].downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )

        self.model.layer2[1].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn1 = nn.BatchNorm2d(64)
        self.model.layer2[1].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn2 = nn.BatchNorm2d(64)

        self.model.layer3[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer3[0].bn1 = nn.BatchNorm2d(128)
        self.model.layer3[0].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[0].bn2 = nn.BatchNorm2d(128)
        self.model.layer3[0].downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )

        self.model.layer3[1].conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn1 = nn.BatchNorm2d(128)
        self.model.layer3[1].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn2 = nn.BatchNorm2d(128)

        self.model.layer4[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer4[0].bn1 = nn.BatchNorm2d(256)
        self.model.layer4[0].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[0].bn2 = nn.BatchNorm2d(256)
        self.model.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )

        self.model.layer4[1].conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn1 = nn.BatchNorm2d(256)
        self.model.layer4[1].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn2 = nn.BatchNorm2d(256)

        self.model.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.model(x)


class QResNet(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = models.resnet18(pretrained=False, num_classes=hidden_dim)
        
        self.model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(32)

        self.model.layer1[0].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[0].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn2 = nn.BatchNorm2d(32)

        self.model.layer1[1].conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn1 = nn.BatchNorm2d(32)
        self.model.layer1[1].conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn2 = nn.BatchNorm2d(32)

        self.model.layer2[0].conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer2[0].bn1 = nn.BatchNorm2d(64)
        self.model.layer2[0].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[0].bn2 = nn.BatchNorm2d(64)
        self.model.layer2[0].downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )

        self.model.layer2[1].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn1 = nn.BatchNorm2d(64)
        self.model.layer2[1].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn2 = nn.BatchNorm2d(64)

        self.model.layer3[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer3[0].bn1 = nn.BatchNorm2d(128)
        self.model.layer3[0].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[0].bn2 = nn.BatchNorm2d(128)
        self.model.layer3[0].downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )

        self.model.layer3[1].conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn1 = nn.BatchNorm2d(128)
        self.model.layer3[1].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn2 = nn.BatchNorm2d(128)

        self.model.layer4[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer4[0].bn1 = nn.BatchNorm2d(256)
        self.model.layer4[0].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[0].bn2 = nn.BatchNorm2d(256)
        self.model.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )

        self.model.layer4[1].conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn1 = nn.BatchNorm2d(256)
        self.model.layer4[1].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn2 = nn.BatchNorm2d(256)

        self.model.fc = nn.Linear(256, hidden_dim)
        
        self.out = nn.Sequential(
            nn.Relu()
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        

    def forward(self, x):
        return self.out(self.model(x))


