import torch.nn.functional as F


def cross_entropy_loss():
    return F.cross_entropy


def calculate_l1_loss(model, loss, lambda_l1=0.001):
    l1_regularization = 0.
    for param in model.parameters():
        l1_regularization += param.abs().sum()
    loss = loss + lambda_l1*l1_regularization
    return loss
