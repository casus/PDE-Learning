import numpy as np
import itertools
from functools import reduce
from utils import lag_diff_mat, to_base, cart_m, powers


class Differentiation:

    def __init__(self, xs, barycentric, dim=1):
        self._xs = xs
        #print(np.max(self._xs[0]),np.max(self._xs[1]))
        self._barycentric = barycentric
        #print(np.max(self._barycentric[0]), np.max(self._barycentric[1]))
        self._dim = dim
        self._diff_mats = [lag_diff_mat(xs[_], barycentric[_]) for _ in range(dim)]
        self._nabla = self.__nabla()

    def get_mats(self):
        return self._diff_mats

    def get_nabla(self):
        return self._nabla

    def _shift_perm(self, i):
        d = self._dim
        perm = np.zeros((d, d))
        for _ in range(d):
            perm[(_+i) % d][_] = 1
        return perm

    def _shift(self, j):
        d = self._dim
        n = len(self._xs[0])

        perm = self._shift_perm(j)
        perm_map = [to_base(_, n) for _ in np.matmul(cart_m(np.arange(0, n), d, flip=False), perm)]

        mat = np.zeros((n**d, n**d))
        for _ in range(n**d):
            mat[_][perm_map[_]] = 1

        return mat

    def __nabla(self):
        d = self._dim
        n = len(self._xs[0])
        dx = [self._dx(j) for j in range(d)]

        nabla = np.zeros((d, n**d, n**d))
        shifts = [self._shift(_+1) for _ in range(d-1)]

        nabla[0] = dx[0]
        for _ in range(d-1):
            nabla[_+1] = np.matmul(np.linalg.inv(shifts[_]), np.matmul(dx[_+1], shifts[_]))

        return nabla

    def _dx(self, j):
        d = self._dim
        n = len(self._xs[0])
        diff_mat = self._diff_mats[j]

        mat = np.zeros((n**d, n**d))
        block_cart = np.array([_ for _ in itertools.product(range(n**(d-1)), range(n), range(n))])
        for _, __, ___ in block_cart:
            mat[_ * n + __][_ * n + ___] = diff_mat[__][___]

        return mat

    def diffs(self, mui):
        d = self._dim
        n = len(self._xs[0])

        diffs = np.zeros((len(mui), n**d, n**d))
        powers_nabla = [powers(self._nabla[_], np.max(mui[:, _])) for _ in range(d)]
        for _, index in enumerate(mui):
            diffs[_] = reduce(np.dot, [powers_nabla[__][index[__]] for __ in range(d)])

        return diffs
