"""
observation_classes.py
======================
Classes for handling observational data, to be used as priors for the CINEMAS analysis.
"""

import numpy as np
import rebound

from . import likelihood


class Observation:
    """A prior specification for an observed parameter."""

    def __init__(
        self,
        distribution: str = "gaussian",
        mean: float | None = None,
        error: float | None = None,
        bounds: tuple[float, float] | None = None,
    ):
        if distribution == "gaussian":
            if mean is None or error is None:
                raise ValueError("Gaussian distribution requires mean and error.")
            if error <= 0:
                raise ValueError("Error must be positive for Gaussian distribution.")
            self.mean = mean
            self.error = error

        elif distribution == "uniform":
            if bounds is None or len(bounds) != 2:
                raise ValueError("Uniform distribution requires bounds.")
            if bounds[1] <= bounds[0]:
                raise ValueError("Uniform bounds must satisfy x_max > x_min.")
            self.bounds = bounds
            self.mean = 0.5 * (bounds[0] + bounds[1])

        else:
            raise ValueError("Unsupported distribution type.")

        self.distribution = distribution


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

        self.minimum_masses = [planet.minimum_mass for planet in planet_observations]
        self.periods = [planet.period for planet in planet_observations]
        self.eccentricities = [planet.eccentricity for planet in planet_observations]

    def plot(self, show_eccentricities=False, omegas=None, **kwargs):
        if show_eccentricities:
            eccentricities = [eccentricity.mean for eccentricity in self.eccentricities]
        else:
            eccentricities = np.zeros(self.n_planets)
        if omegas is None:
            omegas = np.random.uniform(0, 360, size=self.n_planets)

        sim = likelihood.create_rebound_simulations(
            star_mass=self.star_mass.mean,
            masses=[minimum_mass_obs.mean for minimum_mass_obs in self.minimum_masses],
            periods=[period_obs.mean for period_obs in self.periods],
            eccentricities=eccentricities,
            omegas=omegas,
        )

        orbit_plot = rebound.OrbitPlot(sim, **kwargs)
        return orbit_plot.fig
