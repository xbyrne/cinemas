"""
likelihood.py
=============
Likelihood function for the CINEMAS analysis, using SPOCK (in turn using REBOUND).
"""

import numpy as np
from rebound import Simulation
from spock import FeatureClassifier

from . import constants

# =================
# REBOUND functions


def create_single_rebound_simulation(
    star_mass: float,
    masses: np.ndarray,
    periods: np.ndarray,
    eccentricities: np.ndarray = None,
    d_omegas: np.ndarray = None,
) -> Simulation:
    """
    Create a single REBOUND simulation for a given set of orbital parameters.
    Masses should be in Earth masses, and periods in days.
    Eccentricities and omegas are optional, and will be set to 0 if not provided.
    """
    if eccentricities is None:
        eccentricities = np.zeros_like(masses)

    if d_omegas is None:
        d_omegas = np.zeros(len(masses) - 1)  # Omegas are relative to first planet
    omegas = np.concatenate([[0], d_omegas])  # Add the first planet's omega (0)

    sim = Simulation()

    sim.add(m=star_mass)

    for mass, period, ecc, omega in zip(masses, periods, eccentricities, omegas):
        sim.add(m=mass / constants.MSUN_MEARTH, P=period, e=ecc, omega=omega)

    sim.move_to_com()
    return sim



def unpack_theta(theta: np.ndarray):
    """
    Unpack the parameter vector `theta` into its components.
    `theta` should either be of shape (n_parameters,) or (n_samples, n_parameters),
    where n_parameters = 2 + 4 * n_planets (inclination, star mass, minimum masses,
    periods, eccentricities, d_omegas).
    """
    assert theta.ndim in [1, 2], "`theta` should be either 1D or 2D array"

    assert (theta.shape[-1] - 1) % 4 == 0, (
        "`theta` should have 1 + 4 * n_planets parameters: "
        + " (stellar mass, inclination, n_planets*(minimum mass, period, eccentricity),"
        + " (n_planets - 1) * d_omega)."
    )
    n_planets = (theta.shape[-1] - 1) // 4

    inclination = theta[..., 0]
    star_mass = theta[..., 1]
    minimum_masses = theta[..., 2 : 2 + n_planets]
    periods = theta[..., 2 + n_planets : 2 + 2 * n_planets]
    eccentricities = theta[..., 2 + 2 * n_planets : 2 + 3 * n_planets]
    d_omegas = theta[..., 2 + 3 * n_planets :]

    return star_mass, inclination, minimum_masses, periods, eccentricities, d_omegas


# ===================
# Likelihood function


def log_likelihood(
    theta: np.ndarray, spock_classifier: FeatureClassifier = None
) -> float | np.ndarray:
    """
    Log likelihood for the stability of the system, as predicted by SPOCK.
    `theta` must be a 1D array (single parameter set).
    """

    if spock_classifier is None:
        print(
            "Warning: No SPOCK classifier provided; creating a new one."
            " This is inefficient; if you need to call this function multiple times,"
            " consider passing a single classifier."
        )
        spock_classifier = FeatureClassifier()

    star_mass, inclination, minimum_masses, periods, eccentricities, d_omegas = (
        unpack_theta(theta)
    )

    inclination = np.atleast_1d(inclination)

    sim = create_single_rebound_simulation(
        star_mass,
        minimum_masses / np.sin(np.radians(inclination)),
        periods,
        eccentricities,
        d_omegas,
    )

    stability_prob = spock_classifier.predict_stable(sim)
    log_prob = np.log(stability_prob)

    return log_prob
