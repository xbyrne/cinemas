"""
CINEMAS
=======
Constraining INclinations of Exoplanets and their MAsses by Stability
---------------------------------------------------------------------
A Bayesian framework for constraining orbital parameters of exoplanets in
multi-planet systems using dynamical stability as a constraint.
"""

__version__ = "0.1.0.dev0"

from .core import (
    load_system_observations,
    run_mcmc_sampling,
    package_planet_observations,
)

__all__ = [
    "__version__",
    "load_system_observations",
    "package_planet_observations",
    "run_mcmc_sampling",
]
