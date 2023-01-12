import numpy as np
import torch
import itertools
import scipy.special as sps


def to_base(x, bs):
    return int(np.sum([x[_]*bs[_]**(len(x)-_-1) for _ in range(len(x))]))


def cart(xs, flip=True):
    if flip:
        xs = np.flip(xs, 0)
        return np.array(np.flip([_ for _ in itertools.product(*xs)], 1))
    else:
        return np.array([_ for _ in itertools.product(*xs)])


def cart_m(x, m, flip=True):
    prep = np.array([x for _ in range(m)])
    return cart(prep, flip)


def outer(ws):
    is_tensor = torch.is_tensor(ws[0])
    res = ws[-1]
    if is_tensor:
        for _ in range(len(ws)-1):
            res = torch.outer(res, ws[len(ws)-_-2]).reshape(-1)
    else:
        for _ in range(len(ws)-1):
            res = np.outer(res, ws[len(ws)-_-2]).reshape(-1)
    return res


def outer_m(w, m):
    is_tensor = torch.is_tensor(w)
    if m < 1:
        if is_tensor:
            return torch.tensor([])
        else:
            return np.array([])
    res = w
    if is_tensor:
        for _ in range(m-1):
            res = torch.outer(w, res).reshape(-1)
    else:
        for _ in range(m-1):
            res = np.outer(w, res).reshape(-1)
    return res


def outer_arr(arr):
    is_tensor = torch.is_tensor(arr[0])
    n = len(arr)
    if is_tensor:
        res = torch.tensor([1])
        for _ in range(n):
            res = torch.outer(arr[n - (_ + 1)], res).reshape(-1)
    else:
        res = np.array([1])
        for _ in range(n):
            res = np.outer(arr[n - (_ + 1)], res).reshape(-1)
    return res


# please never use
def barycentric(nodes):
    return [1/np.prod([
        (nodes[__] - nodes[_]) for _ in range(len(nodes)) if _ != __
    ]) for __ in range(len(nodes))]


def lag_diff_mat(nodes, w):
    n = len(nodes)
    dx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dx[i][i] = np.sum(
                    [1 / (nodes[i] - nodes[_])
                     for _ in range(n) if _ != i])
            else:
                dx[i][j] = (w[j] / w[i]) * 1 / (nodes[i] - nodes[j])
    return dx


def leja_order(nodes):
    n = len(nodes) - 1
    order = np.arange(1, n+1, dtype=np.int32)
    lj = np.zeros(n + 1, dtype=np.int32)
    lj[0] = 0
    m = 0
    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            p = np.prod([abs(nodes[lj[_]] - nodes[order[i]]) for _ in range(k + 1)])
            if p >= m:
                jj = i
                m = p
        m = 0
        lj[k + 1] = order[jj]
        order = np.delete(order, jj)
    return lj


def mui_lp(m, n, p):
    return np.array([_ for _ in cart_m(np.arange(n+1), m) if np.linalg.norm(_, p) <= n])


def mui_window(d, n, m):
    return np.array([
        _ for _ in cart_m(np.arange(n+1), d) if (n >= np.linalg.norm(_, 1)) & (np.linalg.norm(_, 1) >= m)
    ])


def powers(mat, k):
    n, m = mat.shape
    if n == m:
        pot = np.zeros((k + 1, n, n))
        pot[0] = np.eye(n)
        for i in range(k):
            pot[i + 1] = np.matmul(pot[i], mat)
        return pot
    else:
        return None


def matmul(*mats):
    is_tensor = torch.is_tensor(mats[0])
    n = len(mats)
    r = mats[-1]
    for _ in np.arange(1, n):
        m = mats[n-1-_]
        len_m = m.shape[len(m.shape)-1]
        len_r = len(r)
        q = int(len_r/len_m)
        if len_r == len_m:
            if is_tensor:
                r = torch.matmul(m, r)
            else:
                r = np.matmul(m, r)
        elif len_r % len_m == 0:
            if is_tensor:
                r = torch.cat([torch.matmul(m, r[_*len_m:(_+1)*len_m]) for _ in range(q)])
            else:
                r = np.concatenate([np.matmul(m, r[_*len_m:(_+1)*len_m]) for _ in range(q)])
    return r

def diagmul(w, x):
    d = len(x.shape)
    if d == 1:
        if len(x) == len(w):
            return w*x
        elif len(x) % len(w) == 0:
            return torch.tensor([(w*_).detach().numpy() for _ in torch.split(x, len(w))]).reshape(len(x))
    else:
        d1 = x.shape[0]
        if d1 % len(w) == 0:
            return torch.cat([w for _ in range(int(d1/len(w)))]).reshape(-1, 1)*x


def splitsum(func, x, n):
    if len(x) == n:
        return func(x)
    elif len(x) % n == 0:
        return sum([func(x[_*n:(_+1)*n]) for _ in range(int(len(x)/n))])


def bisection(f, a, b, tol=1e-14):
    assert np.sign(f(a)) != np.sign(f(b))
    m = 0
    while b - a > tol:
        m = a + (b - a) / 2
        fm = f(m)
        if np.sign(f(a)) != np.sign(fm):
            b = m
        else:
            a = m
    return m


def eval_leg_deriv(n, x):
    return (x*sps.eval_legendre(n, x) - sps.eval_legendre(n-1, x))/((x**2-1)/n)


def pullback(nabla, dphi, grid):
    d = len(nabla)
    pb = np.array([
        np.linalg.inv(dphi(_)) for _ in grid])
    nabla_ = np.zeros_like(nabla)
    for _ in range(d):
        pb_ = pb[:, _]
        for __ in range(d):
            nabla_[_] += nabla[__] * pb_[:, __].reshape(-1, 1)
    return nabla_

def det(dphi, grid):
    return np.array([np.linalg.det(dphi(_)) for _ in grid])