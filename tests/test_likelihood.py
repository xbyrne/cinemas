"""
test_likelihood.py
==================
Tests for likelihood and REBOUND helper functions in the CINEMAS package.
"""

import numpy as np
import pytest
from cinemas import constants, likelihood
from pytest import approx


class TestUnpackTheta:
    """Tests for unpacking theta vectors into physical parameters."""

    def test_unpack_theta_on_1d_array(self):
        """Test unpacking a single theta vector."""
        theta = np.array(
            [
                30.0,  # inclination
                1.0,  # star_mass
                2.0,  # minimum masses
                4.0,
                10.0,  # periods
                20.0,
                0.1,  # eccentricities
                0.2,
                0.3,  # omegas
                0.4,
            ]
        )

        star_mass, inclination, minimum_masses, periods, eccentricities, omegas = (
            likelihood.unpack_theta(theta)
        )

        assert star_mass == approx(1.0)
        assert inclination == approx(30.0)
        assert np.allclose(minimum_masses, [2.0, 4.0])
        assert np.allclose(periods, [10.0, 20.0])
        assert np.allclose(eccentricities, [0.1, 0.2])
        assert np.allclose(omegas, [0.3, 0.4])

    def test_unpack_theta_on_2d_array(self):
        """Test unpacking multiple theta vectors."""
        theta = np.array(
            [
                [30.0, 1.0, 2.0, 10.0, 0.1, 0.2],
                [60.0, 1.1, 3.0, 20.0, 0.2, 0.3],
            ]
        )

        star_mass, inclination, minimum_masses, periods, eccentricities, omegas = (
            likelihood.unpack_theta(theta)
        )

        assert np.allclose(star_mass, [1.0, 1.1])
        assert np.allclose(inclination, [30.0, 60.0])
        assert minimum_masses.shape == (2, 1)
        assert periods.shape == (2, 1)
        assert eccentricities.shape == (2, 1)
        assert omegas.shape == (2, 1)

    def test_unpack_theta_raises_for_invalid_ndim(self):
        """Test that unpack_theta rejects arrays with ndim not in [1, 2]."""
        theta = np.zeros((2, 2, 2))

        with pytest.raises(AssertionError, match="1D or 2D"):
            likelihood.unpack_theta(theta)

    def test_unpack_theta_raises_for_invalid_parameter_count(self):
        """Test that unpack_theta rejects arrays that cannot encode full planets."""
        theta = np.array([30.0, 1.0, 2.0])

        with pytest.raises(AssertionError, match=r"2 \+ 4 \* n_planets"):
            likelihood.unpack_theta(theta)


class TestCreateReboundSimulations:
    """Tests for creation of REBOUND simulation objects."""

    def test_create_rebound_simulation_single(self):
        """Test creating a single REBOUND simulation with explicit parameters."""
        sim = likelihood.create_rebound_simulations(
            star_mass=1.0,
            masses=np.array([5.0, 10.0]),
            periods=np.array([10.0, 20.0]),
            eccentricities=np.array([0.1, 0.2]),
            omegas=np.array([0.3, 0.4]),
        )

        assert len(sim.particles) == 3
        assert sim.particles[0].m == approx(1.0)

        assert sim.particles[1].m == approx(5.0 / constants.MSUN_MEARTH)
        assert sim.particles[1].P == approx(10.0)
        assert sim.particles[1].e == approx(0.1)
        assert sim.particles[1].omega == approx(0.3)

        assert sim.particles[2].m == approx(10.0 / constants.MSUN_MEARTH)
        assert sim.particles[2].P == approx(20.0)
        assert sim.particles[2].e == approx(0.2)
        assert sim.particles[2].omega == approx(0.4)

    def test_create_rebound_simulation_sets_default_eccentricity_and_omega(self):
        """Test that eccentricity and omega default to zero when omitted."""
        sim = likelihood.create_rebound_simulations(
            star_mass=1.0,
            masses=np.array([5.0]),
            periods=np.array([10.0]),
        )

        assert len(sim.particles) == 2
        assert sim.particles[1].e == approx(0.0)
        assert sim.particles[1].omega == approx(0.0)

    def test_create_rebound_simulations_multiple(self):
        """Test creating multiple REBOUND simulations in one call."""
        simulations = likelihood.create_rebound_simulations(
            star_mass=np.array([1.0, 1.1]),
            masses=np.array([[5.0], [6.0]]),
            periods=np.array([[10.0], [12.0]]),
            eccentricities=np.array([[0.1], [0.2]]),
            omegas=np.array([[0.3], [0.4]]),
        )

        assert isinstance(simulations, list)
        assert len(simulations) == 2
        assert all(len(sim.particles) == 2 for sim in simulations)
        assert simulations[0].particles[0].m == approx(1.0)
        assert simulations[1].particles[0].m == approx(1.1)


class TestCreateReboundSimulationsFromTheta:
    """Tests for translating theta vectors into REBOUND simulations."""

    def test_create_rebound_simulations_from_theta_passes_true_masses(
        self, monkeypatch
    ):
        """Test that minimum masses are converted to true masses via inclination."""
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
            return "fake-simulation"

        monkeypatch.setattr(
            likelihood, "create_rebound_simulations", fake_create_rebound_simulations
        )

        theta = np.array([30.0, 1.0, 5.0, 10.0, 0.1, 0.2])
        result = likelihood.create_rebound_simulations_from_theta(theta)

        assert result == "fake-simulation"
        assert captured["star_mass"] == approx(1.0)
        assert captured["masses"] == approx(np.array([10.0]))
        assert captured["periods"] == approx(np.array([10.0]))
        assert captured["eccentricities"] == approx(np.array([0.1]))
        assert captured["omegas"] == approx(np.array([0.2]))

    def test_create_rebound_simulations_from_theta_on_multiple_samples(
        self, monkeypatch
    ):
        """Test broadcasting of inclination when theta contains multiple samples."""
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
            return "fake-many"

        monkeypatch.setattr(
            likelihood, "create_rebound_simulations", fake_create_rebound_simulations
        )

        theta = np.array(
            [
                [30.0, 1.0, 5.0, 10.0, 0.1, 0.2],
                [60.0, 1.1, 6.0, 12.0, 0.2, 0.3],
            ]
        )

        result = likelihood.create_rebound_simulations_from_theta(theta)

        assert result == "fake-many"
        assert np.allclose(captured["star_mass"], [1.0, 1.1])
        assert captured["masses"].shape == (2, 1)
        assert np.allclose(
            captured["masses"][:, 0],
            [10.0, 6.0 / np.sin(np.radians(60))],
        )
        assert captured["periods"].shape == (2, 1)


class TestLogLikelihood:
    """Tests for log-likelihood calculation with SPOCK classifier outputs."""

    def test_log_likelihood_with_provided_classifier(self, monkeypatch):
        """
        Test that log_likelihood uses the provided classifier and logs probability.
        """

        class DummyClassifier:
            def predict_stable(self, sims):
                assert sims == "dummy-sims"
                return np.array([0.5, 0.25])

        monkeypatch.setattr(
            likelihood,
            "create_rebound_simulations_from_theta",
            lambda theta: "dummy-sims",
        )

        log_prob = likelihood.log_likelihood(
            theta=np.array([[30.0, 1.0, 5.0, 10.0, 0.1, 0.2]]),
            spock_classifier=DummyClassifier(),
        )

        assert np.allclose(log_prob, np.log([0.5, 0.25]))

    def test_log_likelihood_creates_classifier_if_missing(self, monkeypatch, capsys):
        """Test that log_likelihood constructs a classifier when one is not provided."""

        class DummyClassifier:
            def predict_stable(self, sims):
                assert sims == "dummy-sims"
                return 0.5

        monkeypatch.setattr(
            likelihood,
            "create_rebound_simulations_from_theta",
            lambda theta: "dummy-sims",
        )
        monkeypatch.setattr(likelihood, "FeatureClassifier", DummyClassifier)

        log_prob = likelihood.log_likelihood(
            theta=np.array([30.0, 1.0, 5.0, 10.0, 0.1, 0.2])
        )
        stdout = capsys.readouterr().out

        assert "No SPOCK classifier provided" in stdout
        assert log_prob == approx(np.log(0.5))
