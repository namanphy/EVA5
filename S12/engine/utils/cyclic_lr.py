import numpy as np
import matplotlib.pyplot as plt


class CyclicLR:
    def __init__(self, num_iter, step, lr_min, lr_max):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step = step
        self.iterations = num_iter
        self.lrs = []

    def _calc_cycle(self, iter):
        return np.floor(1 + iter/(2*self.step))

    def cyclic_lr(self, plot=True):
        for iter in range(1, self.iterations+1):
            cycle = self._calc_cycle(iter)

            X = abs(iter/self.step - 2*cycle+1)
            lr = self.lr_min + (self.lr_max - self.lr_min)*(1 - X)

            self.lrs.append(lr)
        if plot:
            self._plot()

    def _plot(self):

        fig = plt.figure(figsize=(10, 3.5))

        plt.title('Cyclic LR Triangular Schedule')
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')

        plt.axhline(self.lr_max, label='max_lr', color='r')
        plt.axhline(self.lr_min, label='min_lr', color='r')

        plt.plot(self.lrs)

        plt.margins(y=0.16)
        plt.tight_layout()
        plt.savefig('clr_plot.png')
