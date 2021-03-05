import torch.optim as optim


def sgd_optimizer(model, lr=0.01, l2_factor=0, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_factor)
