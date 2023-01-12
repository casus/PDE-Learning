import numpy as np
import torch
from cubature import Cubature
from differentiation import Differentiation
from utils import mui_window, diagmul, matmul, splitsum
from scipy.linalg import fractional_matrix_power


class Sobolev(Cubature):

    def __init__(self, s=0, window=0, deg=99, dim=1, quad_type="leg", diffeo=None, quad=None):
        super().__init__(deg, quad_type, dim, quad, diffeo)
        self.diff = Differentiation(self._leja_xs, self._leja_xs_barycentric, dim)
        self._l2_metric = torch.tensor(self._leja_weights.detach().numpy())
        self._inv_l2_metric = 1/self._l2_metric
        self.set_s(s, window)

    def set_s(self, s, window=0):
        self._s = s
        if abs(self._s) < 1:
            self._int_s = 1
        else:
            self._int_s = int(np.floor(abs(self._s)))
        # Implementation of positive fractional order is missing and negative fractional is assuming commutation...
        if (self._s > 0) & (self._int_s > 0):
            self._s = self._int_s
        self._window = window
        if s != 0:
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
            self._dim, self._int_s, self._window
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
        if not self._s == 0:
            m_inv = torch.eye((self._deg+1)**self._dim) + sum([self.l2_adj(_)(_) for _ in self._diffs])
            if self._int_s > 0:
                m = torch.tensor(fractional_matrix_power(m_inv, self._s/self._int_s))
                if torch.is_complex(m):
                    return m.real
                else:
                    return m
            else:
                return m_inv.inverse()

        else:
            return torch.eye(self._deg+1)

    def __weak_m(self):
        if not self._s == 0:
            weak_m_inv = torch.eye((self._deg+1)**self._dim) + sum([self.l2_adj(_)(_, mult="right") for _ in self._diffs])
            if self._int_s > 0:
                weak_m = torch.tensor(fractional_matrix_power(weak_m_inv,self._s/self._int_s))
                if torch.is_complex(weak_m):
                    return weak_m.real
                else:
                    return weak_m
            else:
                return weak_m_inv.inverse()
        else:
            return torch.eye(self._deg+1)

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
        return splitsum(lambda f: self.integral(f**2), u, (self._deg+1)**self._dim)

    def loss(self, u, weak=False):
        if self._s == 0:
            return self.l2_loss(u)
        elif self._s > 0:
            if weak:
                return self.l2_loss(u) + sum([
                    splitsum(lambda f: self.integral(self.l2_adj(_)(f)**2), u, (self._deg+1)**self._dim) for _ in self._diffs
                ])
            else:
                return self.l2_loss(u) + sum([
                    splitsum(lambda f: self.integral(torch.mv(_, f)**2), u, (self._deg+1)**self._dim) for _ in self._diffs
                ])
        else:
            return self.integral(self.m(u, weak=weak)*u)
