import numpy as np


def hyper_rect(*ints):
    midpoints = [(_[1]-_[0])/2 for _ in ints]
    shifts = [(_[1]+_[0])/2 for _ in ints]

    return lambda v: np.array([midpoints[_]*v[_]+shifts[_] for _ in range(len(ints))]),\
        lambda v: np.diag([midpoints[_] for _ in range(len(ints))])


def hyper_sphere(d):
    return lambda v: np.array([sphere_term(v, _) for _ in range(d)]),\
        lambda v: np.array([sphere_term(v, _, grad=True) for _ in range(d)]).T


def prod_sin(v, i, j):
    pi = np.pi
    sin = lambda x: np.sin(x)
    return np.prod(sin(np.array([pi*(v[__]+1) for __ in range(i, j+1)])))


def sphere_term(v, i, grad=False):
    pi = np.pi
    sin = lambda x: np.sin(x)
    cos = lambda x: np.cos(x)

    if i == len(v)-1:
        prod = prod_sin(v, 1, len(v)-1)
        phi = 0.5*(v[0]+1)*prod

        if grad:
            dphi = np.zeros(len(v))
            dphi[0] = 0.5*prod
            for _ in range(1, len(v)):
                dphi[_] = phi*pi*cos(pi*(v[_]+1))/sin(pi*(v[_]+1))

    else:
        prod = prod_sin(v, 1, i)*cos(pi*(v[i+1]+1))
        phi = 0.5*(v[0]+1)*prod

        if grad:
            dphi = np.zeros(len(v))
            dphi[0] = 0.5*prod
            for _ in range(1, len(v)):
                if _ == len(v)-1:
                    dphi[_] = -phi*pi*sin(pi*(v[_]+1))/cos(pi*(v[_]+1))
                else:
                    dphi[_] = phi*pi*cos(pi*(v[_]+1))/sin(pi*(v[_]+1))

    if grad:
        return dphi
    else:
        return phi