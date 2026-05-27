"""
likelihood.py
=============
Likelihood function for the CINEMAS analysis, using SPOCK (in turn using REBOUND).
"""

import numpy as np
from rebound import Simulation
from spock import FeatureClassifier

from . import constants, dataloading

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

    (
        inclination,
        star_mass,
        minimum_masses,
        periods,
        eccentricities,
        longitudes_of_periastron,
        true_anomalies,
    ) = dataloading.unpack_theta(theta)

    inclination = np.atleast_1d(inclination)

    sim = create_rebound_simulation(
        star_mass,
        minimum_masses / np.sin(np.radians(inclination)),
        periods,
        eccentricities,
        longitudes_of_periastron,
        true_anomalies,
    )

    stability_prob = spock_classifier.predict_stable(sim)
    log_prob = np.log(stability_prob)

    return log_prob


# ================
# Helper functions


def create_rebound_simulation(
    star_mass: float,
    masses: np.ndarray,
    periods: np.ndarray,
    eccentricities: np.ndarray = None,
    longitudes_of_periastron: np.ndarray = None,
    true_anomalies: np.ndarray = None,
) -> Simulation:
    """
    Create a single REBOUND simulation for a given set of orbital parameters.
    Masses should be in Earth masses, and periods in days.
    Eccentricities, longitudes of periastron, and true anomalies are optional, and will
    be set to 0 if not provided. WLOG cinemas chooses a reference direction aligned with
    the first planet's periastron, and a reference time such that the first planet's
    true anomaly is 0.
    """
    # e
    if eccentricities is None:
        eccentricities = np.zeros_like(masses)

    # pomega
    if longitudes_of_periastron is None:
        longitudes_of_periastron = np.zeros(len(masses) - 1)
        # Longitudes of periastron are relative to first^^^ planet
    # Add the first planet's longitude of periastron (0)
    longitudes_of_periastron = np.concatenate([[0], longitudes_of_periastron])

    # f
    if true_anomalies is None:
        true_anomalies = np.zeros(len(masses) - 1)
        # True anomalies are relative to first^^^ planet
    true_anomalies = np.concatenate([[0.0], true_anomalies])

    sim = Simulation()

    sim.add(m=star_mass)

    for mass, period, ecc, pomega, f in zip(
        masses, periods, eccentricities, longitudes_of_periastron, true_anomalies
    ):
        sim.add(m=mass / constants.MSUN_MEARTH, P=period, e=ecc, pomega=pomega, f=f)

    sim.move_to_com()
    return sim
