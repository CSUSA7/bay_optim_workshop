"""
A synthetic stand-in for an expensive beam-transport optimization.

This is not SIMION and it is not a real beamline. It is a hand-built test
function that behaves like the real problem in the ways that matter for this
tutorial:

  - it is expensive: one evaluation sleeps for about a minute,
  - it is a black box: you only get the two output numbers back,
  - it is noisy: the outputs are integer counts with Poisson shot noise,
  - the reward is sparse: most input settings return almost zero,
  - the inputs are strongly correlated: only a few combinations of the knobs
    matter, and those combinations are not single knobs.

You give it a vector of normalized electrode voltages in [-1, 1] and it
returns (Nx, Ny): the number of particles counted at the x and y detectors.
The two counts are in tension. Settings that push Nx up tend to pull Ny down
past a point, so there is a genuine trade-off to find. That is what makes the
multi-objective part at the end interesting.

The landscape is fixed by a structure seed, so every student sees the same
problem. The measurement noise uses a separate generator that you control with
the `seed` argument, so runs are reproducible when you want them to be.
"""

import time
import numpy as np


class VirtualBeamline:
    """An expensive, noisy, sparse, two-output objective on a hypercube.

    Parameters
    ----------
    dim : int
        Number of input knobs (normalized voltages). Default 12.
    seconds_per_eval : float
        Wall-clock cost of one evaluation. Default 60.0. Set to 0.0 (or use
        the `fast` flag below) while you are writing and debugging your code.
    noise : bool
        If True, counts are Poisson samples around the mean. If False, you get
        the rounded mean counts (useful for sanity checks, not realistic).
    base_counts : float
        Peak count scale at the best setting.
    structure_seed : int
        Fixes the shape of the landscape. Leave it alone so everyone shares the
        same problem.
    seed : int or None
        Seeds the measurement noise. None means nondeterministic noise.
    """

    def __init__(
        self,
        dim=12,
        seconds_per_eval=60.0,
        noise=True,
        base_counts=1200.0,
        structure_seed=7,
        seed=None,
    ):
        self.dim = int(dim)
        self.seconds_per_eval = float(seconds_per_eval)
        self.noise = bool(noise)
        self.base_counts = float(base_counts)
        self.lower = -np.ones(self.dim)
        self.upper = np.ones(self.dim)
        self.n_evals = 0
        self._rng = np.random.default_rng(seed)
        self._build_structure(structure_seed)

    # ----- internal landscape ------------------------------------------------

    def _build_structure(self, structure_seed):
        g = np.random.default_rng(structure_seed)
        self.n_apertures = 3

        # Build one focus direction plus a few aperture directions as an
        # orthonormal set drawn at random. Each direction is a dense mix of all
        # the knobs, so no single knob means anything on its own and the knobs
        # are strongly correlated in their effect. Only these few combinations
        # move the objective: the effective dimension is much smaller than dim.
        M = g.standard_normal((self.dim, self.n_apertures + 1))
        Q, _ = np.linalg.qr(M)
        self.a_focus = Q[:, 0]
        self.apertures = Q[:, 1:].T  # shape (n_apertures, dim)

        # The beam transmits only when every aperture latent sits near its
        # center at the same time. A product of narrow gates makes that rare:
        # this is the sparse reward. Threading one aperture is easy, threading
        # all of them together is not.
        self.gate_width = 0.16
        self.center_main = np.array([0.5, -0.4, 0.3])

        # A second, easier-to-find pocket that tops out lower than the true
        # ridge. It exists to trap greedy search.
        self.center_decoy = np.array([-0.6, 0.7, -0.2])
        self.decoy_strength = 0.6

        # The two detectors peak at different focus values, which is where the
        # Nx versus Ny trade-off comes from.
        self.focus_peak_x = 0.6
        self.focus_peak_y = -0.5
        self.focus_width = 0.8

    def _gate(self, Z, center):
        """Product of narrow Gaussians across the aperture latents Z."""
        dist2 = np.sum((Z - center) ** 2, axis=1)
        return np.exp(-dist2 / (2 * self.gate_width ** 2))

    def _means(self, V):
        """Noiseless mean counts (Nx, Ny) for one point or a stack of points."""
        V = np.atleast_2d(np.asarray(V, dtype=float))
        z_focus = V @ self.a_focus
        Z = V @ self.apertures.T  # shape (n, n_apertures)

        gate = self._gate(Z, self.center_main) + self.decoy_strength * self._gate(Z, self.center_decoy)

        focus_x = np.exp(-((z_focus - self.focus_peak_x) ** 2) / (2 * self.focus_width ** 2))
        focus_y = np.exp(-((z_focus - self.focus_peak_y) ** 2) / (2 * self.focus_width ** 2))

        mean_x = self.base_counts * gate * focus_x
        mean_y = self.base_counts * gate * focus_y
        return mean_x, mean_y

    # ----- public interface --------------------------------------------------

    def truth(self, V):
        """Return the noiseless mean counts without any time cost.

        Use this for plotting the true landscape and for grading. It is not
        something you are allowed to call inside an optimizer: the real machine
        never hands you the noise-free truth.
        """
        mean_x, mean_y = self._means(V)
        out = np.stack([mean_x, mean_y], axis=-1)
        return out[0] if np.ndim(V) == 1 else out

    def evaluate(self, v, fast=False):
        """Measure one setting and return integer counts (Nx, Ny).

        This is the call that costs you.
        """
        v = np.asarray(v, dtype=float).ravel()
        if v.shape[0] != self.dim:
            raise ValueError(f"expected {self.dim} inputs, got {v.shape[0]}")
        v = np.clip(v, self.lower, self.upper)

        wait = 0.0 if fast else self.seconds_per_eval
        if wait > 0:
            time.sleep(wait)
        self.n_evals += 1

        mean_x, mean_y = self._means(v)
        mean_x, mean_y = float(mean_x[0]), float(mean_y[0])
        if self.noise:
            nx = int(self._rng.poisson(mean_x))
            ny = int(self._rng.poisson(mean_y))
        else:
            nx = int(round(mean_x))
            ny = int(round(mean_y))
        return nx, ny

    def __call__(self, v, fast=False):
        return self.evaluate(v, fast=fast)

    def reset_counter(self):
        self.n_evals = 0

    def random_inputs(self, n=1, seed=None):
        """Uniform random points in the input box, shape (n, dim)."""
        rng = np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper, size=(n, self.dim))
