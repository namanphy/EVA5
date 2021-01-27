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
 

    def forward(self, x):
        return F.log_softmax(x)


def model_summary(model, input_size):
    print(summary(model, input_size=input_size))