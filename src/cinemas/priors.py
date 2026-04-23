"""
priors.py
=========
Prior probability functions for the CINEMAS analysis.
"""

import numpy as np

from . import constants, observation_classes as obs
from .likelihood import unpack_theta


def log_inclination_prior(
    inclination_deg: float | np.ndarray, i_min=constants.I_MIN, i_max=constants.I_MAX
) -> float | np.ndarray:
    """
    Log prior for the inclination of a planet, assuming isotropic orientations.
    """

    log_prior = np.log(np.sin(np.radians(np.clip(inclination_deg, i_min, i_max))))

    if log_prior.ndim == 0:
        if inclination_deg < i_min or inclination_deg > i_max:
            log_prior = -np.inf
    else:
        log_prior[(inclination_deg < i_min) | (inclination_deg > i_max)] = -np.inf

    return log_prior


def log_gaussian_prior(
    x: float | np.ndarray,
    mean: float | np.ndarray,
    error: float | np.ndarray,
    maximum: float | np.ndarray = None,
) -> float | np.ndarray:
    """
    Log prior for a Gaussian distribution; negative values are truncated to -inf.
    The shapes of the inputs should be either:
    - `x`: (1,), (n_planets,), or (n_samples, n_planets)
    - `mean`: (1,), (n_planets,), or (n_planets,)
    - `error`: (1,), (n_planets,), or (n_planets,)
    - `maximum`: (1,), (n_planets,), or (n_planets,) (if provided)
    The output will have the same shape as `x`.
    """
    # Vectorising
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)
    error = np.atleast_1d(error)
    log_prior = -0.5 * ((x - mean) / error) ** 2 - np.log(error * np.sqrt(2 * np.pi))

    log_prior[x < 0] = -np.inf

    if maximum is not None:
        maximum = np.atleast_1d(maximum)
        log_prior[x > maximum] = -np.inf

    return log_prior


def log_uniform_prior(
    x: float | np.ndarray,
    x_min: float,
    x_max: float,
) -> float | np.ndarray:
    """
    Log prior for a uniform distribution; values outside the range are truncated to -inf
    """
    x = np.atleast_1d(x)
    log_prior = -np.log(x_max - x_min) * np.ones_like(x, dtype=float)

    log_prior[(x < x_min) | (x > x_max)] = -np.inf

    return log_prior


def _log_prior_single_parameter(
    x: float | np.ndarray,
    observation: obs.Observation,
    maximum: float | None = None,
) -> float | np.ndarray:
    """
    Log prior for a single parameter, given an Observation object.
    This is a helper function that can be used to compute the log prior for any
    individual parameter, based on its specified distribution and parameters.
    """
    if observation.distribution == "gaussian":
        return log_gaussian_prior(
            x,
            observation.mean,
            observation.error,
            maximum=maximum,
        )
    elif observation.distribution == "uniform":
        return log_uniform_prior(x, observation.bounds[0], observation.bounds[1])

    else:
        raise ValueError(
            f"Unsupported observation distribution: {observation.distribution}"
        )


def log_prior(
    theta: np.ndarray, system_obs: obs.SystemObservations
) -> float | np.ndarray:
    """
    Log prior for the full parameter set.
    `theta` can be either a 1D array (single parameter set) or a 2D array (multiple
    parameter sets; shape (n_samples, n_parameters)).
    """
    star_mass, inclination, minimum_masses, periods, eccentricities, omegas = (
        unpack_theta(theta)
    )

    # Each of the following contributions to the log_prior should either be
    # - a scalar (if `theta` is 1D); or
    # - an array of shape (n_samples,) (if `theta` is 2D).

    # Star mass
    log_p = _log_prior_single_parameter(star_mass, system_obs.star_mass)
    # Inclination
    log_p += log_inclination_prior(inclination)

    # Planetary parameters
    for i in range(system_obs.n_planets):
        # Minimum masses
        log_p += _log_prior_single_parameter(
            minimum_masses[..., i], system_obs.minimum_masses[i]
        )
        # Periods
        log_p += _log_prior_single_parameter(periods[..., i], system_obs.periods[i])
        # Eccentricities
        log_p += _log_prior_single_parameter(
            eccentricities[..., i],
            system_obs.eccentricities[i],
            maximum=1.0,
        )
        # Omegas (uniform between 0 and 360)
        log_p += log_uniform_prior(omegas[..., i], 0, 360)

    return log_p
