"""
test_observation_classes.py
===========================
Test observation data model classes.
"""

import numpy as np
import pytest

from cinemas.observation_classes import (
    Observation,
    PlanetObservations,
    SystemObservations,
)


class TestAccessors:
    """Testing accessors for the observation classes."""

    def test_observation_access(self, simple_observation):
        """Test accessing the mean and error of an Observation."""
        assert simple_observation.distribution == "gaussian"
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

        for observation_list in [
            simple_system_observations.minimum_masses,
            simple_system_observations.periods,
            simple_system_observations.eccentricities,
        ]:
            assert isinstance(observation_list, list)
            assert all(isinstance(item, Observation) for item in observation_list)


class TestSystemObservationsValues:
    """Test that the values in SystemObservations are correct."""

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

    def test_system_observations_unpacked_lists(self, simple_system_observations):
        """SystemObservations should expose lists of Observation objects."""
        assert [
            obs.mean for obs in simple_system_observations.minimum_masses
        ] == [1.0, 2.0]
        assert [obs.mean for obs in simple_system_observations.periods] == [10.0, 20.0]
        assert [
            obs.mean for obs in simple_system_observations.eccentricities
        ] == [0.1, 0.2]

        assert [
            obs.error for obs in simple_system_observations.minimum_masses
        ] == [0.1, 0.2]
        assert [obs.error for obs in simple_system_observations.periods] == [0.5, 1.0]
        assert [
            obs.error for obs in simple_system_observations.eccentricities
        ] == [0.02, 0.04]


class TestUniformObservationSupport:
    """Test support for uniform observation priors."""

    def test_system_observations_uniform_planet_parameter(self):
        planet = PlanetObservations(
            name="b",
            minimum_mass=Observation(distribution="uniform", bounds=(0.5, 1.5)),
            period=Observation(mean=10.0, error=0.5),
            eccentricity=Observation(mean=0.1, error=0.02),
        )
        system_obs = SystemObservations(
            star_name="Star U",
            star_mass=Observation(mean=1.0, error=0.05),
            planet_observations=[planet],
        )

        assert len(system_obs.minimum_masses) == 1
        assert system_obs.minimum_masses[0].distribution == "uniform"
        assert system_obs.minimum_masses[0].bounds == (0.5, 1.5)
        assert np.allclose([system_obs.minimum_masses[0].mean], [1.0])


class TestObservationValidation:
    """Test validation/error branches in Observation initialization."""

    def test_gaussian_requires_mean_and_error(self):
        with pytest.raises(ValueError, match="requires mean and error"):
            Observation(distribution="gaussian", mean=1.0, error=None)

    def test_gaussian_requires_positive_error(self):
        with pytest.raises(ValueError, match="Error must be positive"):
            Observation(distribution="gaussian", mean=1.0, error=0.0)

    def test_uniform_requires_bounds(self):
        with pytest.raises(ValueError, match="requires bounds"):
            Observation(distribution="uniform", bounds=None)

    def test_uniform_requires_increasing_bounds(self):
        with pytest.raises(ValueError, match="x_max > x_min"):
            Observation(distribution="uniform", bounds=(2.0, 2.0))

    def test_unsupported_distribution_raises(self):
        with pytest.raises(ValueError, match="Unsupported distribution type"):
            Observation(distribution="lognormal", mean=1.0, error=0.1)


class TestSystemObservationsPlot:
    """Test plotting branches without relying on external plotting internals."""

    def test_plot_uses_zero_eccentricities_and_random_omegas(
        self,
        simple_system_observations,
        monkeypatch,
    ):
        captured = {}

        def fake_create_rebound_simulations(
            star_mass,
            masses,
            periods,
            eccentricities,
            omegas,
        ):
            captured["star_mass"] = star_mass
            captured["masses"] = masses
            captured["periods"] = periods
            captured["eccentricities"] = eccentricities
            captured["omegas"] = omegas
            return "fake_sim"

        class DummyOrbitPlot:
            def __init__(self, sim, **kwargs):
                captured["sim"] = sim
                captured["kwargs"] = kwargs
                self.fig = "fake_fig"

        monkeypatch.setattr(
            "cinemas.observation_classes.likelihood.create_rebound_simulations",
            fake_create_rebound_simulations,
        )
        monkeypatch.setattr(
            "cinemas.observation_classes.rebound.OrbitPlot",
            DummyOrbitPlot,
        )

        fig = simple_system_observations.plot(show_eccentricities=False)

        assert fig == "fake_fig"
        assert captured["sim"] == "fake_sim"
        assert captured["star_mass"] == 1.0
        assert captured["masses"] == [1.0, 2.0]
        assert captured["periods"] == [10.0, 20.0]
        assert np.allclose(captured["eccentricities"], np.zeros(2))
        assert len(captured["omegas"]) == 2

    def test_plot_uses_observed_eccentricities_and_custom_omegas(
        self,
        simple_system_observations,
        monkeypatch,
    ):
        captured = {}
        custom_omegas = np.array([30.0, 60.0])

        def fake_create_rebound_simulations(
            star_mass,
            masses,
            periods,
            eccentricities,
            omegas,
        ):
            captured["eccentricities"] = eccentricities
            captured["omegas"] = omegas
            return "fake_sim"

        class DummyOrbitPlot:
            def __init__(self, sim, **kwargs):
                self.fig = "fake_fig_2"

        monkeypatch.setattr(
            "cinemas.observation_classes.likelihood.create_rebound_simulations",
            fake_create_rebound_simulations,
        )
        monkeypatch.setattr(
            "cinemas.observation_classes.rebound.OrbitPlot",
            DummyOrbitPlot,
        )

        fig = simple_system_observations.plot(
            show_eccentricities=True,
            omegas=custom_omegas,
            figwidth=5,
        )

        assert fig == "fake_fig_2"
        assert captured["eccentricities"] == [0.1, 0.2]
        assert np.array_equal(captured["omegas"], custom_omegas)
