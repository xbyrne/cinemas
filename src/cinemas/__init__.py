"""
CINEMAS
=======
Constraining INclinations of Exoplanets and their MAsses by Stability
---------------------------------------------------------------------
A Bayesian framework for constraining orbital parameters of exoplanets in
multi-planet systems using dynamical stability as a constraint.
"""

__version__ = "0.4.0.dev0"

from .dataloading import (
    download_multiplanet_systems,
    load_system_observations,
    package_planet_observations,
)
from .mcmc import run_mcmc_sampling

__all__ = [
    "__version__",
    "download_multiplanet_systems",
    "load_system_observations",
    "package_planet_observations",
    "run_mcmc_sampling",
]
