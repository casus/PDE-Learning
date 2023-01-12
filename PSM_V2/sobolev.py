import numpy as np
import torch
from cubature import Cubature
from differentiation import Differentiation
from utils import mui_window, diagmul, matmul, splitsum, barycentric, pullback, det
from scipy.linalg import fractional_matrix_power


class Sobolev(Cubature):

    def __init__(self, s=0, window=0, deg=[100], quad="leg", diffeo=None):
        super().__init__(deg, quad)
        self.size = int(np.prod(self.deg+np.ones_like(self.deg)))

        if diffeo is not None:
            self.phi, self.dphi = diffeo
            self.phi_grid = np.array([self.phi(_) for _ in self.grid])
            self.phi_leja_grid = np.array([self.phi(_) for _ in self.leja_grid])
        else:
            self.phi_grid = self.grid
            self.phi_leja_grid = self.leja_grid

        if quad == "leg":
            leja_axes_barycentric = [
                self.leg_bary(self.axes[_], self.axes_weights[_])[self.axes_leja[_]] for _ in range(self.dim)]
        elif quad == "leglob":
            leja_axes_barycentric = [
                barycentric(self.axes[_])[self.axes_leja[_]] for _ in range(self.dim)]

        self.diff = Differentiation(self.leja_axes, leja_axes_barycentric)

        if diffeo is not None:
            self.diff.nabla = pullback(self.diff.nabla, self.dphi, self.leja_grid)
            #self.leja_weights = det(self.dphi, self.leja_grid)*self.leja_weights

        self.leja_weights = torch.tensor(self.leja_weights)
        self._l2_metric = self.leja_weights
        self._inv_l2_metric = 1/self._l2_metric
        self.set_s(s, window)

    @staticmethod
    def leg_bary(axis, axis_weight):
        return np.array([(-1)**_ for _ in range(len(axis))]) * np.sqrt((1-axis**2)*axis_weight)

    def set_s(self, s, window=0):
        self._s = s
        self._window = window
        self._int_s = int(np.floor(abs(self._s)))

        if self._int_s > 0:
            self._diffs = self.__diffs()
            self._m = self.__m()
            self._weak_m = self.__weak_m()

    def get_diffs(self):
        return self._diffs

    def l2_metric(self):
        return lambda F: diagmul(self._l2_metric, F)

    def metric(self, rev=False, weak=False):
        l2_metric = self._l2_metric
        if (-1)**int(rev)*self._s < 0:
            return lambda F: diagmul(l2_metric, self.m(F, weak=weak))
        else:
            return lambda F: diagmul(l2_metric, self.m_inv(F, weak=weak))

    def __diffs(self):
        return torch.tensor(self.diff.diffs(mui_window(
            self.dim, self._int_s, self._window
        )))

    def l2_adj(self, K):
        l2_metric = self._l2_metric
        l2_inv_metric = self._inv_l2_metric

        def adj(F, mult="left"):
            if mult == "left":
                return diagmul(l2_inv_metric, matmul(K.T, diagmul(l2_metric, F)))
            elif mult == "right":
                return matmul(F, l2_inv_metric.reshape(-1, 1)*K.T*l2_metric)#change

        return adj

    def __m(self):
        if not self._int_s == 0:
            m_inv = torch.eye(self.size) + sum([self.l2_adj(_)(_) for _ in self._diffs])
            return m_inv.inverse()
        else:
            return torch.eye(self.size)

    def __weak_m(self):
        if not self._s == 0:
            weak_m_inv = torch.eye(self.size) + sum([self.l2_adj(_)(_, mult="right") for _ in self._diffs])
            if self._int_s > 0:
                weak_m = torch.tensor(fractional_matrix_power(weak_m_inv,self._s/self._int_s))
                if torch.is_complex(weak_m):
                    return weak_m.real
                else:
                    return weak_m
            else:
                return weak_m_inv.inverse()
        else:
            return torch.eye(self.size)

    def m_inv(self, F, mult="left", weak=False):
        if self._s == 0:
            return F

        if weak:
            if mult == "left":
                return F + sum([matmul(_, self.l2_adj(_)(F)) for _ in self._diffs])
            elif mult == "right":
                return F + sum([self.l2_adj(_)(matmul(F, _), mult) for _ in self._diffs])
        else:
            if mult == "left":
                return F + sum([self.l2_adj(_)(matmul(_, F)) for _ in self._diffs])
            elif mult == "right":
                return F + sum([matmul(F, self.l2_adj(_)(_)) for _ in self._diffs])
        return None

    def m(self, F, mult="left", weak=False):
        if self._s == 0:
            return F

        if weak:
            if mult == "left":
                return matmul(self._weak_m, F)
            elif mult == "right":
                return matmul(F, self._weak_m)
        else:
            if mult == "left":
                return matmul(self._m, F)
            elif mult == "right":
                return matmul(F, self._m)
        return None

    def adjoint(self, K, weak=False):
        adj_K = self.l2_adj(K)
        m = lambda F, mult="left": self.m(F, mult, weak)
        m_inv = lambda F, mult="left": self.m(F, mult, weak)

        def adj(F, mult="left"):
            if self._s == 0:
                return adj_K
            elif self._s > 0:
                if mult == "left":
                    return lambda F: m(adj_K(self.m_inv(F)))
                elif mult == "right":
                    return lambda F: m_inv(adj_K(m(F, mult), mult), mult)
            else:
                if mult == "left":
                    return lambda F: m_inv(adj_K(m(F)))
                elif mult == "right":
                    return lambda F: m(adj_K(m_inv(F, mult), mult), mult)
        return adj

    def l2_loss(self, u):
        return splitsum(lambda f: self.integral(f**2), u, self.size)

    def loss(self, u, weak=False):
        if self._s == 0:
            return self.l2_loss(u)
        elif self._s > 0:
            if weak:
                return self.l2_loss(u) + sum([
                    splitsum(lambda f: self.integral(self.l2_adj(_)(f)**2), u, self.size) for _ in self._diffs
                ])
            else:
                return self.l2_loss(u) + sum([
                    splitsum(lambda f: self.integral(torch.mv(_, f)**2), u, self.size) for _ in self._diffs
                ])
        else:
            return self.integral(self.m(u, weak=weak)*u)
