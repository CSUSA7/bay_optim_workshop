"""Helper code for the Module 5 optimization tutorial.

The pieces students are meant to read and use, not the pieces they build.
The Gaussian process and the acquisition functions are what you write yourself
during the session, so they are not provided here on the main branch.
"""

from .beamline import VirtualBeamline
from .toy import forrester, noisy_forrester, branin
from . import plotting

__all__ = [
    "VirtualBeamline",
    "forrester",
    "noisy_forrester",
    "branin",
    "plotting",
]
