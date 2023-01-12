import numpy as np


def hyper_rect(*intervals):
    midpoints = [(_[1] - _[0])/2 for _ in intervals]
    shifts = [(_[1] + _[0])/2 for _ in intervals]
    phi = lambda x: [midpoints[_]*x + shifts[_] for _ in range(len(intervals))]
    phi.__name__ = "linear"
    det_phi = lambda w: [midpoints[_]*w for _ in range(len(intervals))]

    return phi, det_phi