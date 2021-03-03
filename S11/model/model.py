import sys
import subprocess
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchsummary import summary
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'torchsummary'])
finally:
    from torchsummary import summary


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mp1 = nn.MaxPool2d(2, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        X = F.relu(self.bn1(self.mp1(self.conv1(x))))

        R = F.relu(self.bn2(self.conv2(X)))
        R = F.relu(self.bn3(self.conv3(R)))
        out = X + R
        return out


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = BasicBlock(64, 128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, padding=1, stride=1),
            nn.BatchNorm2d(256))

        self.layer3 = BasicBlock(256, 512)

        self.mp4 = nn.MaxPool2d(4, padding=1)

        self.linear = nn.Linear(512, num_classes)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        out = self.mp4(out)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)  # Using linear instead of F.log_softmax(x).
        # out = F.log_softmax(out)
        return out


def CustomModel():
    return Model(10)


def model_summary(model, input_size):
    print(summary(model, input_size=input_size))