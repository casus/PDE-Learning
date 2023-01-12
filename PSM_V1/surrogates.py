import numpy as np
import torch
from utils import mui_lp, outer_arr
from scipy.special import eval_legendre, eval_chebyt, eval_chebyu


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


class Polynomial(torch.nn.Module):

    def __init__(self, n, p, dim=1, scale=None, poly_type="cheb1"):
        super(Polynomial, self).__init__()
        self._dim = dim
        if scale is None:
            self._scale = torch.ones(dim)
        else:
            self._scale = scale
        self._poly_type = poly_type
        self._mui = mui_lp(dim, n, p)
        self._deg = len(self._mui)
        self.fc = torch.nn.Linear(self._deg, 1, bias=False)
        #

    def get_deg(self):
        return self._deg

    def _data_1d(self, x, deg, scale=1.0):
        poly_type = self._poly_type
        x = np.array(x)/scale
        data = np.zeros((deg+1, len(x)))
        for _ in np.arange(0, deg+1):
            if poly_type == "cheb1":
                data[_] = eval_chebyt(_, x)
            elif poly_type == "cheb2":
                data[_] = eval_chebyu(_, x)
            elif poly_type == "leg":
                data[_] = eval_legendre(_, x)
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
        data_xs = [self._data_1d(__, max_mui, self._scale[_]) for _, __ in enumerate(xs)]
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
        return self.fc(x)#/self._scale


#class ChebyshevNet(torch.nn.Module):