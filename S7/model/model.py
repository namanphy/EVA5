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


class Net(nn.Module):
    def __init__(self, dropout_rate=0.02):
        super(Net, self).__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, bias=False, padding=1), # RF - 3x3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate), # RF - 3x3

            nn.Conv2d(32, 32, 3, bias=False, padding=1), # RF - 5x5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate), # RF - 5x5
        )

        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2), # RF - 10x10
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, bias=False, padding=1), # RF - 12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate), # RF - 12x12

            nn.Conv2d(64, 64, 3, bias=False, padding=1), # RF - 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate), # RF - 14x14
        )

        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2, 2), # RF - 28x28
        )

        self.SepDepthConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1),  # depthwise |  RF - 30x30
            nn.Conv2d(64, 128, kernel_size=1),  # pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
        )

        self.trans3 = nn.Sequential(
            nn.MaxPool2d(2, 2), # RF - 60x60
            # nn.Conv2d(128, 64, kernel_size=1)
        )

        self.AtrousConv = nn.Sequential(
            nn.Conv2d(128, 128, 3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(128, 256, 3, padding=2, dilation=2, bias=False),  # RF - 64x64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
        )

        # self.gap = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1)
        # )

        self.fc = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.SepDepthConv(x)
        x = self.trans3(x)
        x = self.AtrousConv(x)

        x = x.mean(dim=[-2 ,-1]) # GAP Layer
        x = x.view(-1, 256)

        x = self.fc(x)
        return F.log_softmax(x)


def model_summary(model, input_size):
    print(summary(model, input_size=input_size))