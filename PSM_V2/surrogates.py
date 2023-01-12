import numpy as np
import torch
from utils import mui_lp, outer_arr
from scipy.special import eval_chebyt


class NeuralNet(torch.nn.Module):

    def __init__(self, dim=1, activation=lambda x: torch.sin(x), layer_size=1, param_size=50):
        super(NeuralNet, self).__init__()
        self.activation = activation
        self.layer_size = layer_size
        self.param_size = param_size
        self.fc_In = torch.nn.Linear(dim, param_size)
        self.fc_Out = torch.nn.Linear(param_size, 1)
        self.fc_hidden = [torch.nn.Linear(param_size, param_size) for i in range(layer_size)]

    def set_weights(self, val):
        with torch.no_grad():
            self.fc_In.weight.copy_(val*torch.ones_like(self.fc_In.weight))
            self.fc_Out.weight.copy_(val*torch.ones_like(self.fc_Out.weight))
            for _ in range(self.layer_size):
                self.fc_hidden[_].weight.copy_(val*torch.ones_like(self.fc_hidden[_].weight))

    def forward(self, x):
        out = self.fc_In(x)
        out = self.activation(out)
        for layer in self.fc_hidden:
            out = layer(out)
            out = self.activation(out)
        out = self.fc_Out(out)
        return out


class ChebPoly(torch.nn.Module):

    def __init__(self, n, p, dim=1, zero_weights=True):
        super(ChebPoly, self).__init__()
        self._dim = dim
        self._n = n
        self._mui = mui_lp(dim, np.max(n), p)
        self.bound_deg()
        self._deg = len(self._mui)
        self.fc = torch.nn.Linear(self._deg, 1, bias=False)

        if zero_weights:
            self.set_weights_val(0.0)

    def get_deg(self):
        return self._deg

    def bound_deg(self):
        if (self._n is not None) & (len(self._n) == self._dim):
            r = 0
            for i, mui in enumerate(self._mui):
                for j, deg in enumerate(mui):
                    if deg > self._n[j]:
                        self._mui = np.delete(self._mui, i-r, 0)
                        r += 1


    @staticmethod
    def _data_1d(x, deg):
        x = np.array(x)
        data = np.zeros((deg+1, len(x)))
        for _ in np.arange(0, deg+1):
            data[_] = eval_chebyt(_, x)
        return data

    def data_axis(self, x):
        return self.data_axes([x for _ in range(self._dim)])

    def data_axes(self, xs):
        d = len(xs)
        if not d == self._dim:
            return None
        n_xs = [len(_) for _ in xs]
        n_flat = np.prod(n_xs)
        mui = self._mui
        max_mui = int(np.max(mui))
        data_xs = [self._data_1d(__, max_mui) for _, __ in enumerate(xs)]
        data = np.zeros((self._deg, n_flat))
        for i, _ in enumerate(mui):
            data[i] = np.array(outer_arr(np.array([data_xs[(d-1)-__][_[(d-1)-__]] for __ in range(d)])))
        return torch.tensor(data)

    def set_weights_val(self, val):
        with torch.no_grad():
            self.fc.weight.copy_(val * torch.ones_like(self.fc.weight))

    def set_weights(self, w):
        with torch.no_grad():
            self.fc.weight.copy_(w)

    def forward(self, x):
        return self.fc(x)


class Affine(torch.nn.Module):

    def __init__(self, n):
        super(Affine, self).__init__()
        self.fc = torch.nn.Linear(1, n, bias=False)
        self.n = n
        with torch.no_grad():
            self.fc.weight.copy_(torch.linspace(-1.0, 1.0, n).reshape(-1, 1))

    def n(self):
        return self.n

    def forward(self):
        return self.fc.weight


class AffineWrapper:

    def __init__(self, n, tol=10e-4, eps=10e-3):
        self.affine = Affine(n)
        self.tol = tol
        self.eps = eps

    def get_phi(self):
        ints = self.affine().T[0]
        ints, _ = torch.sort(ints)
        midpoints = (ints[1:]-ints[0:-1])/2*(1+self.eps)
        shifts = (ints[1:]+ints[0:-1])/2

        def phi(x):
            if torch.is_tensor(x):
                return midpoints.reshape(-1, 1)*x +\
                       shifts.reshape(-1, 1)*torch.ones_like(x)
            else:
                return midpoints.reshape(-1, 1).detach().numpy()*x +\
                       shifts.reshape(-1, 1).detach().numpy()*np.ones_like(x)

        return phi

    def follow_up(self):
        with torch.no_grad():
            ints = self.affine().T[0]
            ints, _ = torch.sort(ints)
            delta = torch.abs(ints[1:]-ints[0:-1])
            w = torch.tensor([ints[0]], requires_grad=False)
            for i, _ in enumerate(delta):
                if _-2*self.eps > self.tol:
                    w = torch.cat((w, torch.tensor([ints[i + 1]])), dim=0)
            self.affine = Affine(len(w))
            max_ = torch.max(w)
            min_ = torch.min(w)
            mid = 2 / (max_ - min_)
            shift = -1 - mid * min_
            transfo = mid*w + torch.ones_like(w)*shift
            self.affine.fc.weight.copy_(transfo.reshape(-1, 1))
