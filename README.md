# Module 5: optimizing an expensive black box

A hands-on tutorial where you write a Bayesian optimizer from scratch and then
use it on a hard problem. Part of Day 2 of the MIT and UNAL workshop on
experimental design and digital twins.

Yesterday you ran SIMION from Python and plotted detector counts against one
voltage. Here the machine has twelve voltages at once, every measurement is
slow, and most settings give nothing. You will build the kind of optimizer that
real beamlines use to tune themselves, understand every line of it, and then
scale it up with a library.

## Setup

You need Python 3.9 or newer. From the repository root:

```
pip install -r requirements.txt
pip install -e .
jupyter lab
```

The second line installs the small `blackbox` helper package so the notebooks
can import it from anywhere. If you skip it the notebooks still work, because
each one adds the repository to the path in its first cell.

Quick check, run from the repository root:

```
python -c "from blackbox import VirtualBeamline; print(VirtualBeamline().dim, 'knobs')"
```

Then open the `notebooks` folder and start with notebook 1.

## What you will do

1. **The expensive problem.** Meet the black box, feel the cost of a
   measurement, and see why blind search is hopeless.
2. **A Gaussian process from scratch.** Write the kernel and the posterior in
   `blackbox/gp.py`, and watch the model report honest uncertainty.
3. **The Bayesian optimization loop.** Write an acquisition function, assemble
   the loop, and watch it reach good solutions in far fewer measurements than
   random search.
4. **Scaling up with Optuna.** Hand the heavy lifting to a library, attack the
   twelve-knob problem, and find the trade-off between the two detectors.

Work in groups. The notebooks have cells marked *Discuss*; stop and talk those
through before moving on. The code you write goes in the `TODO` cells and in the
two open methods of `blackbox/gp.py`.

## Solutions

The worked solutions, with the code filled in and extra explanation, live on the
`sols` branch. Try it yourself first. When you want to compare:

```
git checkout sols
```

and `git checkout main` to come back.

## Layout

```
blackbox/
  beamline.py    the expensive twelve-knob black box (a stand-in, not SIMION)
  gp.py          the Gaussian process you fill in
  toy.py         small cheap functions for the early notebooks
  plotting.py    plot helpers so the notebooks stay readable
notebooks/       the four parts, in order
```

## A note on the black box

`VirtualBeamline` is a synthetic function, not SIMION and not real hardware. It
was built to behave like the real transport problem in the ways that matter: it
is slow, noisy, mostly empty, its knobs are strongly correlated, and its two
outputs are in tension. The point is to learn the method on something that
behaves like the real thing without burning real beam time.
