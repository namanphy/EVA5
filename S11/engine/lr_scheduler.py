from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR


def step_lr_scheduler(optimizer, step_size=None, gamma=0.15):
    if step_size is None:
        raise Exception('step size value must be provided with valid integer')
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def reduce_lr_on_plateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False, min_lr=0):
    return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr)


def once_cycle_lr(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, div_factor=25.0):
    return OneCycleLR(optimizer, max_lr, total_steps=total_steps, epochs=epochs, steps_per_epoch=steps_per_epoch,
                      pct_start=pct_start, div_factor=div_factor, final_div_factor=10000.0, last_epoch=-1,
                      verbose=False)
