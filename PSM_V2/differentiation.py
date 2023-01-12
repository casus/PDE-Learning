import numpy as np
import itertools
from functools import reduce
from utils import lag_diff_mat, to_base, cart, powers


class Differentiation:

    def __init__(self, axes, barycentric):
        self._dim = len(axes)
        self._sizes = [len(_) for _ in axes]
        self._size = int(np.prod(self._sizes))
        self._axes = axes
        self._barycentric = barycentric
        self._diff_mats = [lag_diff_mat(axes[_], barycentric[_]) for _ in range(self._dim)]
        self.nabla = self.__nabla()

    def _shift_perm(self, i):
        d = self._dim
        perm = np.zeros((d, d))
        for _ in range(d):
            perm[(_+i) % d][_] = 1
        return perm

    def _shift(self, j):
        szs = self._sizes
        sz = self._size

        perm = self._shift_perm(j)
        iterator = np.matmul(cart([np.arange(0, _) for _ in szs], flip=False), perm)
        perm_map = [to_base(_, szs) for _ in iterator]

        mat = np.zeros((sz, sz))
        for _ in range(sz):
            mat[_][perm_map[_]] = 1

        return mat

    def __nabla(self):
        d = self._dim
        sz = self._size
        dx = [self._dx(j) for j in range(d)]

        nabla = np.zeros((d, sz, sz))
        shifts = [self._shift(_+1) for _ in range(d-1)]

        nabla[0] = dx[0]
        for _ in range(d-1):
            nabla[_+1] = np.matmul(np.linalg.inv(shifts[_]), np.matmul(dx[_+1], shifts[_]))

        return nabla
    #???
    def _dx(self, j):
        d = self._dim
        n = self._sizes[j]
        sz = self._size
        diff_mat = self._diff_mats[j]

        mat = np.zeros((sz, sz))
        block_cart = np.array([_ for _ in itertools.product(range(int(sz/n)), range(n), range(n))])
        for _, __, ___ in block_cart:
            mat[_ * n + __][_ * n + ___] = diff_mat[__][___]

        return mat

    def diffs(self, mui):
        d = self._dim
        sz = self._size

        diffs = np.zeros((len(mui), sz, sz))
        powers_nabla = [powers(self.nabla[_], np.max(mui[:, _])) for _ in range(d)]
        for _, index in enumerate(mui):
            diffs[_] = reduce(np.dot, [powers_nabla[__][index[__]] for __ in range(d)])

        return diffs
