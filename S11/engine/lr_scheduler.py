from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def step_lr_scheduler(optimizer, step_size=None, gamma=0.15):
    if step_size is None:
        raise Exception('step size value must be provided with valid integer')
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def reduce_lr_on_plateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False, min_lr=0):
    return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr)
