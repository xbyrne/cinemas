"""
test_observation_classes.py
===========================
Test observation data model classes.
"""

import numpy as np

from cinemas.observation_classes import Observation, PlanetObservations


class TestAccessors:
    """Testing accessors for the observation classes."""

    def test_observation_access(self, simple_observation):
        """Test accessing the mean and error of an Observation."""
        assert simple_observation.mean is not None
        assert simple_observation.error is not None

    def test_planet_observations_access(self, simple_planet_observations):
        """Test accessing attributes of PlanetObservations."""
        assert simple_planet_observations.name is not None
        assert simple_planet_observations.minimum_mass.mean is not None
        assert simple_planet_observations.period.mean is not None
        assert simple_planet_observations.eccentricity.mean is not None

    def test_system_observations_access(self, simple_system_observations):
        """Test accessing attributes of SystemObservations."""
        assert simple_system_observations.star_name is not None

        assert isinstance(simple_system_observations.star_mass, Observation)

        assert isinstance(simple_system_observations.planet_observations, list)
        for planet_obs in simple_system_observations.planet_observations:
            assert isinstance(planet_obs, PlanetObservations)

        assert isinstance(simple_system_observations.n_planets, int)
        assert simple_system_observations.n_planets == len(
            simple_system_observations.planet_observations
        )

        for unpacked_array in [
            simple_system_observations.minimum_masses,
            simple_system_observations.periods,
            simple_system_observations.eccentricities,
            simple_system_observations.minimum_masses_errors,
            simple_system_observations.periods_errors,
            simple_system_observations.eccentricities_errors,
        ]:
            assert isinstance(unpacked_array, np.ndarray)


class TestSystemObservationsValues:
    """Test that the values in SystemObservations are correct."""

    def test_observation_values(self, simple_observation):
        """Test that the values in an Observation are correct."""
        assert simple_observation.mean == 1.0
        assert simple_observation.error == 0.1

    def test_planet_observations_values(self, simple_planet_observations):
        """Test that the values in PlanetObservations are correct."""
        assert simple_planet_observations.name == "Planet b"
        assert simple_planet_observations.minimum_mass.mean == 1.0
        assert simple_planet_observations.minimum_mass.error == 0.1
        assert simple_planet_observations.period.mean == 10.0
        assert simple_planet_observations.period.error == 0.5
        assert simple_planet_observations.eccentricity.mean == 0.1
        assert simple_planet_observations.eccentricity.error == 0.02

    def test_system_observations_values(self, simple_system_observations):
        """Test that the values in SystemObservations are correct."""
        assert simple_system_observations.star_name == "Star A"
        assert simple_system_observations.star_mass.mean == 1.0
        assert simple_system_observations.star_mass.error == 0.05

        planet_b = simple_system_observations.planet_observations[0]
        assert planet_b.name == "b"
        assert planet_b.minimum_mass.mean == 1.0
        assert planet_b.minimum_mass.error == 0.1
        assert planet_b.period.mean == 10.0
        assert planet_b.period.error == 0.5
        assert planet_b.eccentricity.mean == 0.1
        assert planet_b.eccentricity.error == 0.02

        planet_c = simple_system_observations.planet_observations[1]
        assert planet_c.name == "c"
        assert planet_c.minimum_mass.mean == 2.0
        assert planet_c.minimum_mass.error == 0.2
        assert planet_c.period.mean == 20.0
        assert planet_c.period.error == 1.0
        assert planet_c.eccentricity.mean == 0.2
        assert planet_c.eccentricity.error == 0.04

    def test_system_observations_unpacked_arrays(self, simple_system_observations):
        """Test that the unpacked arrays in SystemObservations have correct values."""
        assert np.allclose(simple_system_observations.minimum_masses, [1.0, 2.0])
        assert np.allclose(simple_system_observations.periods, [10.0, 20.0])
        assert np.allclose(simple_system_observations.eccentricities, [0.1, 0.2])

        assert np.allclose(simple_system_observations.minimum_masses_errors, [0.1, 0.2])
        assert np.allclose(simple_system_observations.periods_errors, [0.5, 1.0])
        assert np.allclose(
            simple_system_observations.eccentricities_errors, [0.02, 0.04]
        )
