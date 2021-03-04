import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel


class ResidualBlock(BaseModel):

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class BasicBlock(BaseModel):

    def __init__(self, in_planes, planes, res_block=None):
        super(BasicBlock, self).__init__()

        self.res_block = None

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(planes)

        if res_block is not None:
            self.res_block = nn.Sequential(
                res_block(planes, planes)
            )

    def forward(self, x):
        X = F.relu(self.bn1(self.mp1(self.conv1(x))))
        if self.res_block is not None:
            X = X + self.res_block(X)
        return X


class ResModel(BaseModel):
    def __init__(self, num_classes, in_planes):
        super(ResModel, self).__init__()

        self.intro_layer = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = BasicBlock(64, 128, res_block=ResidualBlock)
        self.layer2 = BasicBlock(128, 256)
        self.layer3 = BasicBlock(256, 512, res_block=ResidualBlock)

        self.maxpool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.intro_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)  # Using linear instead of F.log_softmax(x).
        return out
