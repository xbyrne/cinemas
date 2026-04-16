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
    x: float | np.ndarray, mean: float | np.ndarray, error: float | np.ndarray
) -> float | np.ndarray:
    """
    Log prior for a Gaussian distribution; negative values are truncated to -inf.
    The shapes of the inputs should be either:
    - `x`: (1,), (n_planets,), or (n_samples, n_planets)
    - `mean`: (1,), (n_planets,), or (n_planets,)
    - `error`: (1,), (n_planets,), or (n_planets,)
    The output will have the same shape as `x`.
    """
    # Vectorising
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)
    error = np.atleast_1d(error)

    log_prior = -0.5 * ((x - mean) / error) ** 2 - np.log(error * np.sqrt(2 * np.pi))

    log_prior[x < 0] = -np.inf

    return log_prior


def log_uniform_prior(x: float | np.ndarray, x_min: float, x_max: float) -> float:
    """
    Log prior for a uniform distribution; values outside the range are truncated to -inf
    """
    log_prior = -np.log(x_max - x_min) * np.ones_like(x)

    log_prior[(x < x_min) | (x > x_max)] = -np.inf

    return log_prior


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
    log_p = log_gaussian_prior(
        star_mass, system_obs.star_mass.mean, system_obs.star_mass.error
    )
    # Inclination
    log_p += log_inclination_prior(inclination)
    # Minimum masses
    log_p += log_gaussian_prior(
        minimum_masses, system_obs.minimum_masses, system_obs.minimum_masses_errors
    ).sum(axis=-1)
    # Periods
    log_p += log_gaussian_prior(
        periods, system_obs.periods, system_obs.periods_errors
    ).sum(axis=-1)
    # Eccentricities
    log_p += log_gaussian_prior(
        eccentricities,
        system_obs.eccentricities,
        system_obs.eccentricities_errors,
    ).sum(axis=-1)
    # Omegas (uniform between 0 and 360)
    log_p += log_uniform_prior(omegas, 0, 360).sum(axis=-1)

    return log_p
