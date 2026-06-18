"""
A Gaussian process you build yourself.

This file is the core exercise of part 2. Two methods are left for you to write:
the kernel and the posterior. The notebook for part 2 walks you through both,
with the formulas and the array shapes you need. Everything else is here so you
can test your work as you go.

When you finish a method, save this file. If your notebook has autoreload on
(the setup cell turns it on), the change takes effect on the next cell you run.
"""

import numpy as np


class GaussianProcess:
    """A Gaussian process with a squared-exponential (RBF) kernel.

    Parameters
    ----------
    lengthscale : float
        How far apart two inputs can be before the model treats them as
        unrelated. Small means wiggly, large means smooth.
    signal_var : float
        The variance of the function, that is, how far it swings from its mean.
    noise_var : float
        The variance of the observation noise. Keep a tiny value even for
        noise-free data, so the matrix stays invertible.
    """

    def __init__(self, lengthscale=0.2, signal_var=1.0, noise_var=1e-6):
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.X = None
        self.y = None

    @staticmethod
    def _as_2d(X):
        """Treat a 1-D array as a column of n one-dimensional points."""
        X = np.asarray(X, dtype=float)
        return X[:, None] if X.ndim == 1 else X

    def kernel(self, A, B):
        """Covariance between every row of A and every row of B.

        A has shape (n, d), B has shape (m, d). The result has shape (n, m).

        The squared-exponential kernel is
            k(a, b) = signal_var * exp(-0.5 * ||a - b||^2 / lengthscale^2).

        A clean way to get all the pairwise squared distances without a loop:
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b
        Build that as an (n, m) array, clip any tiny negative values to zero,
        then apply the exponential.
        """
        A = self._as_2d(A)
        B = self._as_2d(B)
        # TODO: return the (len(A), len(B)) RBF covariance matrix.
        raise NotImplementedError("Write the RBF kernel. See the docstring.")

    def fit(self, X, y):
        """Store the data and factor the training covariance once.

        This is given. It calls your kernel, so it starts working as soon as
        your kernel is correct. We use a Cholesky factor instead of a raw
        inverse because it is faster and far more stable numerically.
        """
        self.X = self._as_2d(X)
        self.y = np.asarray(y, dtype=float).ravel()
        K = self.kernel(self.X, self.X) + self.noise_var * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def posterior(self, X_test):
        """Posterior mean and standard deviation at the test points.

        X_test has shape (m, d). Return (mean, std), each of length m.

        With K already factored in fit() as self.L (lower triangular) and
        self.alpha = K^{-1} y, the posterior at test points X* is

            K_s   = kernel(X_train, X_test)          shape (n, m)
            mean  = K_s^T . alpha                     length m
            v     = solve(L, K_s)                     shape (n, m)
            var   = signal_var - sum(v * v, axis=0)   length m
            std   = sqrt(max(var, small_positive))

        The diagonal of kernel(X_test, X_test) equals signal_var for the RBF
        kernel, which is why the variance line is so short.
        """
        X_test = self._as_2d(X_test)
        # TODO: return (mean, std) using self.L and self.alpha. See the docstring.
        raise NotImplementedError("Write the posterior. See the docstring.")

    # ----- given helpers -----------------------------------------------------

    def sample_prior(self, X_test, n_samples=3, rng=None):
        """Draw whole functions from the prior at the test points.

        Given. It only needs your kernel. Each row of the result is one sampled
        function evaluated at X_test.
        """
        rng = np.random.default_rng() if rng is None else rng
        X_test = self._as_2d(X_test)
        K = self.kernel(X_test, X_test) + 1e-9 * np.eye(len(X_test))
        L = np.linalg.cholesky(K)
        return (L @ rng.standard_normal((len(X_test), n_samples))).T

    def sample_posterior(self, X_test, n_samples=3, rng=None):
        """Draw whole functions from the posterior. Given; needs kernel and fit."""
        rng = np.random.default_rng() if rng is None else rng
        X_test = self._as_2d(X_test)
        K_s = self.kernel(self.X, X_test)
        mean = K_s.T @ self.alpha
        v = np.linalg.solve(self.L, K_s)
        cov = self.kernel(X_test, X_test) - v.T @ v + 1e-9 * np.eye(len(X_test))
        L = np.linalg.cholesky(cov)
        return mean[None, :] + (L @ rng.standard_normal((len(X_test), n_samples))).T

    def log_marginal_likelihood(self):
        """Bonus. The evidence for the current hyperparameters.

        Useful for choosing the lengthscale by maximizing it. The formula is
            -0.5 * y^T alpha - sum(log(diag(L))) - 0.5 * n * log(2 pi).
        Delete the raise and return that once you want to try the bonus.
        """
        # TODO (bonus): return the log marginal likelihood.
        raise NotImplementedError("Bonus: write the log marginal likelihood.")
