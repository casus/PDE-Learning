import numpy as np
import torch
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style


mpl.rcParams['lines.linewidth'] = 0.8
style.use("seaborn-pastel")


class Benchmark:

    def __init__(self, model, gt, axes, data=None):
        self.model = model
        self._gt = gt
        self.set_axes(axes, data)
        self._losses = np.array([])
        self._time = 0.0
        self.eval_model = None
        self.eval_gt = None

    def set_gt(self, gt):
        self._gt = gt

    def set_axes(self, axes, data=None):
        self._axes = axes

        if data is None:
            if len(axes) == 1:
                self._data = torch.tensor(axes[0].reshape(-1, 1))
            elif len(axes) > 1:
                self._data = torch.tensor([_ for _ in itertools.product(*axes)])
        else:
            self._data = data

    def get_losses(self):
        return self._losses

    def get_time(self):
        return self._time

    def reset_time(self):
        self._time = 0

    def set_data(self):
        return self._data

    def eval(self):
        self.eval_model = self.model(self._data)
        self.eval_model = self.eval_model.detach().numpy().T[0].reshape([len(_) for _ in self._axes])

        if self._gt is not None:
            self.eval_gt = self._gt(*np.meshgrid(*self._axes))
            return self.eval_model, self.eval_gt

        return self.eval_model

    def _plot1d(self, data, title="", file_name=None):
        return self.plot1d(data, self._axes[0], title, file_name)

    @staticmethod
    def plot1d(data, x, title="", file_name=None, ticks=5):
        xticks = [_ for _ in range(len(x)) if _ % np.round(len(x) / (ticks - 1)) == 0]

        fig = mpl.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(200)
        plt.xticks(xticks, np.round(x[xticks], decimals=2), fontsize=8)
        plt.yticks(fontsize=8)
        plt.plot(data)
        plt.grid(linestyle='--')
        plt.title(title, fontsize=8)

        if file_name is not None:
            fig.savefig(file_name+'.png', dpi=75)

        plt.show()

    def _plot2d(self, data, title="", file_name=None):
        return self.plot2d(data, self._axes[0], self._axes[1], title, file_name)

    @staticmethod
    def plot2d(data, x, y, title="", file_name=None, ticks=5):
        xticks = [_ for _ in range(len(x)) if _ % np.round(len(x) / (ticks - 1)) == 0]
        yticks = [_ for _ in range(len(y)) if _ % np.round(len(y) / (ticks - 1)) == 0]

        fig = mpl.pyplot.gcf()
        fig.set_size_inches(12.33, 7.0)
        fig.set_dpi(200)
        plt.xticks(xticks, np.round(x[xticks], decimals=2), fontsize=8)
        plt.yticks(yticks, np.round(y[yticks], decimals=2), fontsize=8)
        plt.imshow(data, cmap="Spectral", origin="lower")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.grid(linestyle='--')
        plt.title(title, fontsize=8)
        #plt.clim(-0.4, 0.4) #additionally set this

        if file_name is not None:
            fig.savefig(file_name+'.png', dpi=75)

        plt.show()

    def plot_losses(self, lower=0, upper=-1, suffix=None):
        if (lower <= upper) & upper <= len(self._losses):
            title = "Losses Plot"
            fn = None
            if suffix is not None:
                fn = "losses_plot_" + suffix

            self._plot1d(self._losses[lower:upper], title=title, file_name=fn)

    def plot_model(self, suffix=None):
        if self.eval_model is not None:
            title = "Model Plot"
            fn = None
            if suffix is not None:
                fn = "model_plot_" + suffix

            dim = len(self._axes)
            if dim == 1:
                self._plot1d(self.eval_model, title=title, file_name=fn)
            elif dim == 2:
                self._plot2d(self.eval_model, title=title, file_name=fn)

    def plot_gt(self, suffix=None):
        if self.eval_model is not None:
            title = "Ground Truth Plot"
            fn = None
            if suffix is not None:
                fn = "gt_plot_" + suffix

            dim = len(self._axes)
            if dim == 1:
                self._plot1d(self.eval_gt, title=title, file_name=fn)
            elif dim == 2:
                self._plot2d(self.eval_gt, title=title, file_name=fn)

    def plot_abs_err(self, suffix=None):
        if (self.eval_gt is not None) & (self.eval_model is not None):
            title = "Absolute Error Plot"
            fn = None

            if suffix is not None:
                fn = "abs_err_plot_" + suffix

            data = abs(self.eval_gt-self.eval_model)

            dim = len(self._axes)
            if dim == 1:
                self._plot1d(data, title=title, file_name=fn)
            elif dim == 2:
                self._plot2d(data, title=title, file_name=fn)

    def lp_err(self, p=np.inf):
        if (self.eval_gt is not None) & (self.eval_model is not None):
            if p == np.inf:
                return np.max(abs(self.eval_gt-self.eval_model))
            else:
                return np.mean((abs(self.eval_gt-self.eval_model))**p)**(1/p)
