import numpy as np
import torch
from scipy.special import roots_legendre, roots_chebyt
from utils import cart, outer, leja_order, barycentric


class Cubature:

    def __init__(self, deg=99, quad_type="leg", dim=1, quad=None, diffeo=None):
        self._deg = deg
        self._dim = dim

        if quad is None:
            self._grid_1d, self._weights_1d, self._barycentric = getattr(self, quad_type)()
            leja_order_1d = leja_order(self._grid_1d)
            self._leja_grid_1d = self._grid_1d[leja_order_1d]
            self._leja_weights_1d = self._weights_1d[leja_order_1d]
            self._leja_barycentric = self._barycentric[leja_order_1d]
        else:
            self._grid_1d, self._weights_1d, self._barycentric,\
                self._leja_grid_1d, self._leja_weights_1d, self._leja_barycentric = quad

        # Apply Diffeomorphisms

        phi = lambda x: [x for _ in range(dim)]
        det_phi = lambda w: [w for _ in range(dim)]
        phi.__name__ = "linear"

        if diffeo is not None:
            phi, det_phi = diffeo

        self._xs = phi(self._grid_1d)
        self._leja_xs = phi(self._leja_grid_1d)
        self._leja_xs_weights = det_phi(self._leja_weights_1d)

        if phi.__name__ == "linear":
            self._leja_xs_barycentric = [self._leja_barycentric for _ in range(dim)]
        else:
            self._leja_xs_barycentric = [barycentric(_) for _ in self._leja_xs]

        self._grid = cart(self._xs)
        self._leja_grid = torch.tensor(cart(self._leja_xs))
        self._leja_weights = torch.tensor(outer(self._leja_xs_weights))

    def get_quad(self):
        return self._grid_1d, self._weights_1d, self._barycentric, \
               self._leja_grid_1d, self._leja_weights_1d, self._leja_barycentric

    def get_xs(self):
        return self._xs, self._leja_xs

    def get_grid(self):
        return self._grid

    def get_leja_grid(self):
        return self._leja_grid

    def get_leja_weights(self):
        return self._leja_weights

    def leg(self):
        grid, weights = roots_legendre(self._deg+1)
        barycentric = np.array([(-1)**_ for _ in range(len(grid))])*np.sqrt((1-grid**2)*weights)
        return grid, weights, barycentric

    def cheb1(self):
        grid, weights = roots_chebyt(self._deg+1)
        barycentric = np.array([(-1)**_*np.sin((2*_+1)/(2*self._deg+2)*np.pi) for _ in range(self._deg+1)])
        return grid, weights, barycentric

    def integral(self, u):
        return torch.sum(self._leja_weights*u)
