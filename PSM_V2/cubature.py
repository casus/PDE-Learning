import numpy as np
import torch
import scipy.special as sps
from functools import partial
from utils import cart, outer, leja_order, bisection, eval_leg_deriv


class Cubature:

    def __init__(self, deg=[100], quad="leg"):
        self.dim = len(deg)
        self.deg = deg
        self.quad = quad

        self.axes, self.axes_weights, self.axes_leja = self._axes()
        self.leja_axes, self.leja_axes_weights = self._leja_axes()

        self.grid = cart(self.axes)
        self.leja_grid = cart(self.leja_axes)
        self.leja_weights = outer(self.leja_axes_weights)

    def _axes(self):
        axes = []
        axes_weights = []
        axes_leja = []

        for i, _ in enumerate(self.deg):
            if _ in self.deg[0:i]:
                j = self.deg.index(_)

                axis = axes[j]
                axis_weight = axes_weights[j]
                axis_leja = axes_leja[j]

            else:
                axis, axis_weight = getattr(self, self.quad)(_)
                axis_leja = leja_order(axis)

            axes.append(axis)
            axes_weights.append(axis_weight)
            axes_leja.append(axis_leja)

        return axes, axes_weights, axes_leja

    def _leja_axes(self):
        return [self.axes[_][self.axes_leja[_]] for _ in range(self.dim)], \
               [self.axes_weights[_][self.axes_leja[_]] for _ in range(self.dim)], \

    @staticmethod
    def leg(deg):
        axis, axis_weight = sps.roots_legendre(deg+1)
        return axis, axis_weight

    #https://relate.cs.illinois.edu/course/CS450-S20/f/demos/upload/quadrature_and_diff/Gaussian%20quadrature%20weight%20finder.html
    #explicit rule for lobatto barycentric weights in order to reduce numerical error
    #but additionally implement sth which can handle barycentric(nodes)
    @staticmethod
    def leglob(deg):
        n = deg+1
        brackets = sps.legendre(n-1).weights[:, 0]
        axis = np.zeros(n)
        axis[0] = -1
        axis[-1] = 1

        for _ in range(n-2):
            axis[_+1] = bisection(
                partial(eval_leg_deriv, n-1),
                brackets[_], brackets[_+1])

        axis_weight = np.zeros(n)
        axis_weight[0] = 2/n/(n-1)
        axis_weight[-1] = 2/n/(n-1)

        for _ in range(n-2):
            axis_weight[_+1] = 2/n/(n-1)/(sps.eval_legendre(n-1, axis[_+1]))**2

        return axis, axis_weight

    def integral(self, u):
        if torch.is_tensor(self.leja_weights):
            return torch.sum(self.leja_weights*u)
        else:
            return torch.sum(torch.tensor(self.leja_weights)*u)
