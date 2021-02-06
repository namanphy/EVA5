import sys
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
try:
    from torchsummary import summary
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'torchsummary'])
finally:
    from torchsummary import summary


class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        self.layer1 = "Current Date/Time: ", datetime.now()

        self.x1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, bias=False, padding=1),  # i/p - 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.x3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x4 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # i/p - 16x16
        )

        self.x5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x8 = nn.Sequential(
            nn.MaxPool2d(2, 2), # RF - 8x8
        )

        self.x9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x10 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1),  # i/p - 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x12 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

        self.x13 = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x):
        print(self.layer1)

        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x1 + x2)
        x4 = self.x4(x1 + x2 + x3)

        x5 = self.x5(x4)
        x6 = self.x6(x4 + x5)
        x7 = self.x7(x4 + x5 + x6)
        x8 = self.x8(x5 + x6 + x7)

        x9 = self.x9(x8)
        x10 = self.x10(x8 + x9)
        x11 = self.x11(x8 + x9 + x10)

        x12 = self.x12(x11)
        x12 = x12.view(-1, 64)

        out = self.x13(x12)

        return F.log_softmax(out)


def model_summary(model, input_size):
    print(summary(model, input_size=input_size))
