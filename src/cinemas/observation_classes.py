"""
observation_classes.py
======================
Classes for handling observational data, to be used as priors for the CINEMAS analysis.
"""

import numpy as np
import rebound

from . import likelihood


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

    def plot(self, show_eccentricities=False, omegas=None, **kwargs):
        if show_eccentricities:
            eccentricities = self.eccentricities
        else:
            eccentricities = np.zeros_like(self.eccentricities)
        if omegas is None:
            omegas = np.random.uniform(0, 360, size=self.n_planets)

        sim = likelihood.create_rebound_simulations(
            star_mass=self.star_mass.mean,
            masses=self.minimum_masses,
            periods=self.periods,
            eccentricities=eccentricities,
            omegas=omegas,
        )

        orbit_plot = rebound.OrbitPlot(sim, **kwargs)
        return orbit_plot.fig
