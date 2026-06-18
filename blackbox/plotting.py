"""
Plot helpers so the notebooks stay focused on ideas instead of matplotlib.

Nothing here is required to understand the method. It is here so a student can
write one line and get a clear picture of what their model and their
acquisition function are doing.
"""

import numpy as np
import matplotlib.pyplot as plt

# A small, consistent palette used across the notebooks.
DATA = "#222222"      # observed points
MODEL = "#2E5A87"     # posterior mean
BAND = "#2E5A87"      # uncertainty band (drawn with low alpha)
TRUTH = "#A31F34"     # the true function, when we are allowed to show it
ACQ = "#E37222"       # acquisition function
PICK = "#2E7D32"      # the next point the acquisition chooses


def use_clean_style():
    """Set readable defaults. Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.3),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "font.size": 11,
    })


def plot_gp_1d(x, mean, std, X_obs, y_obs, truth=None, draws=None, ax=None,
               title=None, n_std=2.0):
    """Draw a 1D posterior: mean, an n_std band, the data, and optionally the
    true function and a few sampled functions.

    x is the dense grid, mean and std are the posterior at that grid, X_obs and
    y_obs are the observed points.
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.ravel(x)
    mean = np.ravel(mean)
    std = np.ravel(std)

    if draws is not None:
        for d in np.atleast_2d(draws):
            ax.plot(x, d, color=MODEL, alpha=0.25, lw=1.0)
    if truth is not None:
        ax.plot(x, np.ravel(truth), color=TRUTH, lw=2.0, label="true function")

    ax.fill_between(x, mean - n_std * std, mean + n_std * std,
                    color=BAND, alpha=0.18, label=f"+/- {n_std:g} std")
    ax.plot(x, mean, color=MODEL, lw=2.0, label="posterior mean")
    ax.scatter(np.ravel(X_obs), np.ravel(y_obs), color=DATA, zorder=5,
               s=40, label="observations")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_acquisition(x, acq, x_next=None, ax=None, label="acquisition"):
    """Draw an acquisition curve and mark the point it selects."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 2.2))
    x = np.ravel(x)
    acq = np.ravel(acq)
    ax.plot(x, acq, color=ACQ, label=label)
    ax.fill_between(x, np.zeros_like(acq), acq, color=ACQ, alpha=0.15)
    if x_next is not None:
        ax.axvline(float(x_next), color=PICK, ls="--", lw=2.0, label="next point")
    ax.set_xlabel("x")
    ax.set_ylabel(label)
    ax.legend(loc="best")
    return ax


def plot_convergence(best_values, ax=None, label=None, marker="o"):
    """Best value found so far against the number of evaluations."""
    if ax is None:
        _, ax = plt.subplots()
    best = np.ravel(best_values)
    ax.plot(np.arange(1, len(best) + 1), best, marker=marker, label=label)
    ax.set_xlabel("number of evaluations")
    ax.set_ylabel("best value so far")
    if label:
        ax.legend(loc="best")
    return ax


def plot_pareto(objectives, ax=None, label=None, color=MODEL, show_front=True):
    """Scatter two-objective results and outline the non-dominated front.

    objectives has shape (n, 2). This assumes you are maximizing both columns,
    which matches (Nx, Ny) counts.
    """
    if ax is None:
        _, ax = plt.subplots()
    F = np.atleast_2d(np.asarray(objectives, dtype=float))
    ax.scatter(F[:, 0], F[:, 1], color=color, alpha=0.5, s=30, label=label)

    if show_front and len(F) > 1:
        front = _pareto_front_max(F)
        order = np.argsort(front[:, 0])
        ax.plot(front[order, 0], front[order, 1], color=TRUTH, lw=2.0,
                marker="o", label="Pareto front")
    ax.set_xlabel("Nx")
    ax.set_ylabel("Ny")
    ax.legend(loc="best")
    return ax


def pareto_front(objectives):
    """Non-dominated rows of an (n, 2) array, assuming both columns maximized."""
    return _pareto_front_max(np.atleast_2d(np.asarray(objectives, dtype=float)))


def _pareto_front_max(F):
    """Return the non-dominated rows of F, assuming maximization of both cols.

    A row is kept when no other row beats it on both objectives at once.
    """
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        beats_i = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        if np.any(beats_i):
            keep[i] = False
    return F[keep]
