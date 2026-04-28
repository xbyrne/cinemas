# CINEMAS :clapper:

**Constraining INclinations of Exoplanets and their MAsses by Stability**

[![PyPI version](https://badge.fury.io/py/cinemas.svg)](https://badge.fury.io/py/cinemas)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![CINEMAS](https://raw.githubusercontent.com/xbyrne/cinemas/main/absolute_cinemas.png)

CINEMAS is a Bayesian framework for constraining the inclinations, and hence the true masses, of exoplanets in compact multi-planet systems detected with the radial velocity (RV) method.

The true mass $M$ of an exoplanet is not measured with the RV method, only the minimum mass $M_{\rm min}=M\sin i$, where $i$ is the (generally unknown) inclination angle. However, if $i$ were too low, this would mean the true masses of the planets ($M_j=M_{{\rm min}, j} / \sin i$) would be so big that the system would be dynamically unstable.

Assuming isotropy, the prior probability distribution on $i$ is $\pi(i)=\sin i$. The probability that a compact system with a given inclination (and hence given masses) is dynamically stable can be calculated quickly using the [`spock`](https://github.com/dtamayo/spock/) package. CINEMAS uses MCMC to calculate posterior distributions for the inclination, and thus the true masses of these exoplanets in inclined multi-planet systems. The result? Absolute cinema(s).


## Installation

Install CINEMAS directly from PyPI:

```bash
pip install cinemas
```

Or, for development, clone the repository and install in editable mode:

```bash
git clone https://github.com/xbyrne/cinemas
cd cinemas
pip install -e ".[dev]"
```

## Quickstart

### Loading observational data

Observational data is loaded from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/index.html), using their [TAP service](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html). It is then loaded into a `SystemObservations` object, which includes all the observational data needed for a prior on the configuration parameters.

```python
import cinemas

# Load observational constraints for a system
catalogue = cinemas.download_multiplanet_systems()
system_obs = cinemas.load_system_observations(
    star_name="Barnard's star",
    exoplanet_catalogue=catalogue
)
```

### Running MCMC sampling

```python
import cinemas

# Run MCMC to constrain the inclination and true masses
samples, tau, acceptance_fraction = cinemas.run_mcmc_sampling(
    system_obs,
    nsteps=5000,
    nwalkers=None  # By default, 2*number of params
)

print(f"Autocorrelation time: {tau}")
print(f"Acceptance fraction: {acceptance_fraction}")
```


## References

- [SPOCK: Stability of Planetary Orbital Configurations Klassifier](https://github.com/dtamayo/spock/)
- [emcee: The MCMC Hammer](https://github.com/dfm/emcee)
- [REBOUND: N-body simulations](https://github.com/hannorein/rebound)
