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


def create_rebound_simulations(
    star_mass: float | np.ndarray,
    masses: np.ndarray,
    periods: np.ndarray,
    eccentricities: np.ndarray = None,
    omegas: np.ndarray = None,
) -> Simulation | list[Simulation]:
    """
    Create REBOUND simulations for a given set of orbital parameters.
    Masses should be in Earth masses, and periods in days.
    Eccentricities and omegas are optional, and will be set to 0 if not provided.
    """
    if eccentricities is None:
        eccentricities = np.zeros_like(masses)
    if omegas is None:
        omegas = np.zeros_like(masses)

    if (isinstance(star_mass, np.ndarray) and star_mass.ndim == 1) or isinstance(
        star_mass, list
    ):
        # Multiple simulations
        simulations = []

        for star_mass_val, mass, period, ecc, omega in zip(
            star_mass, masses, periods, eccentricities, omegas
        ):
            # Recursion ftw
            simulations.append(
                create_rebound_simulations(star_mass_val, mass, period, ecc, omega)
            )

        return simulations

    # Single simulation
    sim = Simulation()

    sim.add(m=star_mass)

    for mass, period, ecc, omega in zip(masses, periods, eccentricities, omegas):
        sim.add(m=mass / constants.MSUN_MEARTH, P=period, e=ecc, omega=omega)

    sim.move_to_com()
    return sim


def create_rebound_simulations_from_theta(
    theta: np.ndarray,
) -> Simulation | list[Simulation]:
    """
    Creates REBOUND simulation(s) from the parameter vector `theta`.
    The shape of `theta` should be either (n_parameters,) or (n_samples, n_parameters).
    """

    star_mass, inclination, minimum_masses, periods, eccentricities, omegas = (
        unpack_theta(theta)
    )

    # minimum_masses are either of shape (n_planets,) or (n_samples, n_planets);
    # inclination is either of shape (1,) or (n_samples,).
    # We want the true masses, which are minimum_masses / sin(inclination).
    # To do this, we need to ensure that the shapes are compatible for broadcasting.
    if inclination.ndim == 1 and minimum_masses.ndim == 2:
        inclination = inclination[:, None]

    masses = minimum_masses / np.sin(np.radians(inclination))

    return create_rebound_simulations(
        star_mass, masses, periods, eccentricities, omegas
    )


def unpack_theta(theta: np.ndarray):
    """
    Unpack the parameter vector `theta` into its components.
    `theta` should either be of shape (n_parameters,) or (n_samples, n_parameters),
    where n_parameters = 2 + 4 * n_planets (inclination, star mass, minimum masses,
    periods, eccentricities, omegas).
    """
    assert theta.ndim in [1, 2], "`theta` should be either 1D or 2D array"

    assert (
        theta.shape[-1] - 2
    ) % 4 == 0, "`theta` should have 2 + 4 * n_planets parameters"
    n_planets = (theta.shape[-1] - 2) // 4

    inclination = theta[..., 0]
    star_mass = theta[..., 1]
    minimum_masses = theta[..., 2 : 2 + n_planets]
    periods = theta[..., 2 + n_planets : 2 + 2 * n_planets]
    eccentricities = theta[..., 2 + 2 * n_planets : 2 + 3 * n_planets]
    omegas = theta[..., 2 + 3 * n_planets : 2 + 4 * n_planets]

    return star_mass, inclination, minimum_masses, periods, eccentricities, omegas


# ===================
# Likelihood function


def log_likelihood(
    theta: np.ndarray, spock_classifier: FeatureClassifier = None
) -> float | np.ndarray:
    """
    Log likelihood for the stability of the system, as predicted by SPOCK.
    `theta` can be either a 1D array (single parameter set) or a 2D array (multiple
    parameter sets; shape (n_samples, n_parameters)).
    """

    if spock_classifier is None:
        print(
            "Warning: No SPOCK classifier provided; creating a new one."
            " This is inefficient; if you need to call this function multiple times,"
            " consider passing a single classifier."
        )
        spock_classifier = FeatureClassifier()

    sims = create_rebound_simulations_from_theta(theta)

    stability_prob = spock_classifier.predict_stable(sims)
    log_prob = np.log(stability_prob)

    return log_prob
