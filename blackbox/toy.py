"""
Small, cheap, plottable functions for the early part of the tutorial.

You cannot see a Gaussian process do its job in twelve dimensions. So we learn
the machinery on a 1D function you can draw on a single axis, then a 2D one,
and only then move to the expensive twelve-knob beamline.

Everything here is instant and deterministic. The expensive, noisy object lives
in beamline.py.
"""

import numpy as np


def forrester(x):
    """The Forrester function on [0, 1] (Forrester, Sobester, Keane, 2008).

    A standard one-dimensional test function for Bayesian optimization. It has
    one global minimum near x = 0.757, a shallow local minimum on the left, and
    enough curvature that a naive search is easy to fool. We treat it as a
    minimization problem.

    Accepts a scalar or an array and returns the same shape.
    """
    x = np.asarray(x, dtype=float)
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


forrester.bounds = (0.0, 1.0)
forrester.argmin = 0.75724876


def noisy_forrester(x, sigma=0.5, rng=None):
    """Forrester plus Gaussian observation noise, to practice the noisy case."""
    rng = np.random.default_rng() if rng is None else rng
    y = forrester(x)
    return y + rng.normal(0.0, sigma, size=np.shape(y))


def branin(xy):
    """The Branin-Hoo function, a classic 2D minimization test.

    Defined on x in [-5, 10], y in [0, 15]. Three global minima of equal depth.
    Pass a point as a length-2 array, or a stack of points with shape (n, 2).
    """
    p = np.atleast_2d(np.asarray(xy, dtype=float))
    x, y = p[:, 0], p[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    val = a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s
    return val[0] if np.ndim(xy) == 1 else val


branin.bounds = [(-5.0, 10.0), (0.0, 15.0)]
