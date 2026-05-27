"""
nested_sampling.py
==================
Functions for running nested sampling using dynesty for the CINEMAS package.
"""

import numpy as np
from dynesty import NestedSampler
from dynesty.pool import Pool
from scipy.stats import norm, truncnorm
from spock import FeatureClassifier

from . import constants, likelihood, observation_classes as obs


def run_nested_sampling(
    system_obs: obs.SystemObservations,
    nlive: int = 500,
    dlogz: float = 0.5,
    checkpoint_file: str | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Run nested sampling using dynesty to obtain posterior samples for the system
    parameters.

    Parameters
    ----------
    system_obs : obs.SystemObservations
        The system observations object containing priors and constraints.
    nlive : int, optional
        Number of live points for nested sampling (default: 500).
    dlogz : float, optional
        Tolerance on log evidence; sampling stops when contribution falls below
        this (default: 0.5).
    checkpoint_file : str | None, optional
        Path to the checkpoint file for resuming sampling (default: None).
    """

    if checkpoint_file is not None:
        checkpoint_every = 600  # Save checkpoint every 10 minutes
    else:
        checkpoint_every = None

    n_params = 5 * system_obs.n_planets
    periodic_indices = list(range(2 + 3 * system_obs.n_planets, n_params))
    # ^including the longitudes of periastron and true anomalies, which are periodic

    spock_classifier = FeatureClassifier()

    with Pool(
        10,
        likelihood.log_likelihood,
        prior_transform,
        logl_args=(spock_classifier,),
        ptform_args=(system_obs,),
    ) as pool:
        sampler = NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=n_params,
            nlive=nlive,
            periodic=periodic_indices,
            pool=pool,
        )

        sampler.run_nested(
            dlogz=dlogz,
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
        )

    results = sampler.results
    results.summary()  # Print summary of results to console

    return results


def prior_transform(u: np.ndarray, system_obs: obs.SystemObservations) -> np.ndarray:
    """
    Transform unit cube [0,1]^d to the prior space.

    Parameters
    ----------
    u : np.ndarray
        Unit cube coordinates, shape (batch_size, n_parameters).
    system_obs : obs.SystemObservations
        System observations containing prior specifications.

    Returns
    -------
    np.ndarray
        Transformed parameters in the prior space.
    """

    n_planets = system_obs.n_planets
    n_params = 5 * n_planets

    theta = np.zeros((n_params,))

    # Inclinations
    cos_imin = np.cos(np.radians(constants.I_MIN))
    cos_imax = np.cos(np.radians(constants.I_MAX))
    theta[0] = np.degrees(np.arccos(cos_imin - (cos_imin - cos_imax) * u[0]))

    # Stellar mass: use observation distribution
    theta[1] = _transform_observation(u[1], system_obs.star_mass, clip=(0, np.inf))
    # Planet parameters
    for i in range(n_planets):
        # Minimum mass
        theta[2 + i] = _transform_observation(
            u[2 + i], system_obs.minimum_masses[i], clip=(0.01, np.inf)
        )
        # Period
        theta[2 + n_planets + i] = _transform_observation(
            u[2 + n_planets + i], system_obs.periods[i], clip=(0.001, np.inf)
        )
        # Eccentricity
        theta[2 + 2 * n_planets + i] = _transform_observation(
            u[2 + 2 * n_planets + i], system_obs.eccentricities[i], clip=(0, 0.999)
        )
        # Relative longitudes of periastron: uniform [0, 360]
        if i < n_planets - 1:
            theta[2 + 3 * n_planets + i] = 360 * u[2 + 3 * n_planets + i]

        # True anomalies: planets 2..n are uniform [0, 360]
        if i < n_planets - 1:
            theta[1 + 4 * n_planets + i] = 360 * u[1 + 4 * n_planets + i]

    return theta


def _transform_observation(
    u: np.ndarray, observation: obs.Observation, clip: tuple = (None, None)
) -> np.ndarray:
    """
    Transform unit cube [0, 1] to an observation's prior distribution.

    Parameters
    ----------
    u : np.ndarray
        Unit cube coordinates, shape (batch_size,).
    observation : obs.Observation
        Observation object specifying the prior distribution.
    clip : tuple, optional
        (min, max) values to clip the transformed parameters to (default: no clipping).

    Returns
    -------
    np.ndarray
        Transformed values, shape (batch_size,).
    """
    if observation.distribution == "gaussian":
        # Use inverse error function to transform to Gaussian
        # (with clipping if necessary)
        if clip != (None, None):
            a, b = (
                (clip[0] - observation.mean) / observation.error,
                (clip[1] - observation.mean) / observation.error,
            )
            return truncnorm.ppf(
                u, a=a, b=b, loc=observation.mean, scale=observation.error
            )
        else:
            return norm.ppf(u, loc=observation.mean, scale=observation.error)

    elif observation.distribution == "uniform":
        # Uniform: straightforward linear transform
        return observation.bounds[0] + u * (
            observation.bounds[1] - observation.bounds[0]
        )

    else:
        raise ValueError(f"Unsupported distribution: {observation.distribution}")
