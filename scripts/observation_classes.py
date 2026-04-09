"""
observation_classes.py
======================
Classes for handling observational data, to be used as priors for the CRIMES analysis.
"""

import numpy as np


class Observation:
    """An observation with a mean and asymmetric error bars."""

    def __init__(self, mean: float, error: float):
        self.mean = mean
        self.error = error


class PlanetObservations:
    """Collected observations for a single planet."""

    def __init__(
        self,
        name: str,
        minimum_mass: Observation,
        period: Observation,
        eccentricity: Observation,
    ):
        self.name = name
        self.minimum_mass = minimum_mass
        self.period = period
        self.eccentricity = eccentricity


class SystemObservations:
    """
    Collected observations for a planetary system, including the stellar mass, and a
    list of planet observations.
    """

    def __init__(
        self,
        star_name: str,
        star_mass: Observation,
        planet_observations: list[PlanetObservations],
    ):
        self.star_name = star_name
        self.star_mass = star_mass
        self.planet_observations = planet_observations
        self.n_planets = len(planet_observations)

        self.minimum_masses = np.array(
            [planet.minimum_mass.mean for planet in planet_observations]
        )
        self.periods = np.array([planet.period.mean for planet in planet_observations])
        self.eccentricities = np.array(
            [planet.eccentricity.mean for planet in planet_observations]
        )

        self.minimum_masses_errors = np.array(
            [planet.minimum_mass.error for planet in planet_observations]
        )
        self.periods_errors = np.array(
            [planet.period.error for planet in planet_observations]
        )
        self.eccentricities_errors = np.array(
            [planet.eccentricity.error for planet in planet_observations]
        )
