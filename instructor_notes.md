# Instructor notes (Module 5)

Private notes for running the session. Not meant for students.

## Shape of the session (about 90 minutes)

- Part 1, the expensive problem: 10 to 15 min. Mostly running and discussing.
- Part 2, the Gaussian process: 30 to 35 min. This is the real work. Most groups
  spend the time on the kernel broadcasting and the posterior.
- Part 3, the BO loop: 25 to 30 min. The acquisition function is short; the loop
  is where they assemble it.
- Part 4, Optuna: 15 min. Light coding, mostly seeing the payoff and the bridge
  to Module 6.

If time is short, Part 4 can be demoed from the `sols` branch.

## How the black box works

`VirtualBeamline` takes 12 voltages in [-1, 1]. Internally it projects them onto
four dense, random, orthonormal directions: one focus direction and three
aperture directions. Only these combinations matter, so the 12 knobs are
strongly correlated and the effective dimension is four.

- **Sparse reward.** The beam transmits only when all three aperture latents sit
  near their centers at once, which is a product of three narrow Gaussians. Most
  of the space gives zero. The median total counts over random settings is 0.
- **Decoy.** There is a second, easier pocket that tops out around 60 percent of
  the true peak. Greedy search can park there.
- **Two outputs in tension.** The x and y detectors peak at different focus
  values, so raising Nx past a point lowers Ny. That is the Pareto trade-off in
  Part 4 and the analogue of transmission versus emittance from the lecture.
- **Noise.** Counts are Poisson around the mean.
- **Cost.** Each call sleeps `seconds_per_eval` (default 60). `fast=True` skips
  the wait. The notebooks use fast mode and ask students to count evaluations as
  if each still cost a minute.

The landscape is fixed by `structure_seed=7`. Leave it alone so every group sees
the same problem. `truth(v)` gives the noise free mean and is used only for
plots and grading; it is not something an optimizer is allowed to call.

To make it harder or easier, the knobs to turn are `gate_width` (smaller is
sparser, default 0.16), `dim`, `decoy_strength`, and `base_counts`.

## Where students get stuck

- **Kernel shapes.** They write a loop or get an (n,) instead of an (n, m)
  array. Point them at the identity in the docstring and the broadcasting with
  `[:, None]` and `[None, :]`.
- **Posterior.** Forgetting to use the Cholesky factor and trying to invert K
  directly. It still works on these tiny problems, but steer them to the given
  `solve(L, ...)` form.
- **Expected improvement.** Dividing by a zero standard deviation. The guard is
  in the hint.
- **Optuna objective.** Returning a single number for the multi-objective study,
  or forgetting `fast=True`.

One thing to expect, not a bug: in Part 4 the TPE and random curves cross. Random
is briefly ahead of TPE near trial 100 before TPE pulls clearly ahead by 150
(about 1600 counts versus 1050). That crossover is normal and is good material
for the Discuss prompt about parking in a mediocre region.

## The bridge to Module 6

The last cell of Part 4 collects every trial into a table of voltages mapped to
(Nx, Ny). That table is exactly the training set for the emulator in Module 6.
It is worth pausing here: the search and the emulator are two halves of one pipeline.

## Regenerating the notebooks

The notebooks are generated from `build/gen.py` (not tracked, kept locally).
`python build/gen.py` writes the student notebooks; `python build/gen.py
--solution` writes the worked ones. The `sols` branch holds the solution
notebooks and the filled in `blackbox/gp.py`.
