import torch.nn as nn
from torchsummary import summary


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def summary(self, input_size):
        print(summary(self, input_size=input_size))
