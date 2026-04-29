"""
test_priors.py
==============
Tests for prior probability functions in the CINEMAS package.
"""

import numpy as np
from cinemas import priors
from cinemas.observation_classes import (
    Observation,
    PlanetObservations,
    SystemObservations,
)
from pytest import approx


class TestInclinationPrior:
    """Tests for inclination prior behavior and bounds."""

    def test_log_inclination_prior_scalar_inside_bounds(self):
        """Inclinations within bounds should return finite log prior."""
        value = priors.log_inclination_prior(30.0)

        assert np.isfinite(value)
        assert value == approx(np.log(np.sin(np.radians(30.0))))

    def test_log_inclination_prior_scalar_outside_bounds(self):
        """Inclinations outside bounds should return -inf."""
        assert priors.log_inclination_prior(0.01) == -np.inf
        assert priors.log_inclination_prior(91.0) == -np.inf

    def test_log_inclination_prior_vector_mixed_values(self):
        """Vector inputs should apply truncation element-wise."""
        inclination = np.array([10.0, 30.0, 95.0])
        log_p = priors.log_inclination_prior(inclination)

        assert log_p.shape == (3,)
        assert np.isfinite(log_p[0])
        assert np.isfinite(log_p[1])
        assert log_p[2] == -np.inf


class TestGaussianPrior:
    """Tests for Gaussian prior behavior."""

    def test_log_gaussian_prior_matches_analytic_value(self):
        """Gaussian prior should match the standard log-PDF."""
        x = np.array([1.0])
        mean = 1.0
        error = 0.1

        log_p = priors.log_gaussian_prior(x, mean, error)
        expected = -0.5 * ((x - mean) / error) ** 2 - np.log(error * np.sqrt(2 * np.pi))

        assert np.allclose(log_p, expected)

    def test_log_gaussian_prior_truncates_negative_values(self):
        """Negative values should be assigned -inf."""
        x = np.array([0.5, -0.1, 1.2])
        log_p = priors.log_gaussian_prior(x, mean=1.0, error=0.2)

        assert np.isfinite(log_p[0])
        assert log_p[1] == -np.inf
        assert np.isfinite(log_p[2])


class TestUniformPrior:
    """Tests for uniform prior behavior."""

    def test_log_uniform_prior_inside_and_outside(self):
        """Uniform prior should be constant inside bounds and -inf outside."""
        x = np.array([-1.0, 10.0, 200.0, 500.0])
        log_p = priors.log_uniform_prior(x, x_min=0.0, x_max=360.0)

        assert log_p[0] == -np.inf
        assert log_p[1] == approx(-np.log(360.0))
        assert log_p[2] == approx(-np.log(360.0))
        assert log_p[3] == -np.inf

    def test_log_uniform_prior_scalar_input(self):
        """Uniform prior should support scalar inputs without indexing errors."""
        log_p = priors.log_uniform_prior(10.0, x_min=0.0, x_max=360.0)

        assert np.shape(log_p) == (1,)
        assert log_p[0] == approx(-np.log(360.0))


class TestFullPrior:
    """Tests for the full prior over theta."""

    def test_log_prior_on_single_theta_returns_finite(self, simple_system_observations):
        """A physically plausible theta should yield finite total log prior."""
        theta = np.array(
            [
                45.0,
                1.0,
                1.0,
                2.0,
                10.0,
                20.0,
                0.1,
                0.2,
                180.0,
                200.0,
            ]
        )

        log_p = priors.log_prior(theta, simple_system_observations)

        assert np.shape(log_p) == (1,)
        assert np.isfinite(log_p[0])

    def test_log_prior_on_multiple_theta_preserves_sample_axis(
        self, simple_system_observations
    ):
        """2D theta input should return one log-prior value per sample."""
        theta = np.array(
            [
                [45.0, 1.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0],
                [50.0, 1.1, 1.2, 2.1, 10.1, 19.9, 0.15, 0.25, 100.0, 250.0],
            ]
        )

        log_p = priors.log_prior(theta, simple_system_observations)

        assert log_p.shape == (2,)
        assert np.all(np.isfinite(log_p))

    def test_log_prior_rejects_unphysical_parameters(self, simple_system_observations):
        """Out-of-range inclination should produce -inf prior."""
        theta = np.array(
            [
                0.01,
                1.0,
                1.0,
                2.0,
                10.0,
                20.0,
                0.1,
                0.2,
                180.0,
                200.0,
            ]
        )

        log_p = priors.log_prior(theta, simple_system_observations)

        assert log_p[0] == -np.inf

    def test_log_prior_accepts_uniform_planet_priors(self):
        """Uniform priors in SystemObservations should be consumed by log_prior."""
        system_obs = SystemObservations(
            star_name="Star U",
            star_mass=Observation(mean=1.0, error=0.05),
            planet_observations=[
                PlanetObservations(
                    name="b",
                    minimum_mass=Observation(distribution="uniform", bounds=(0.5, 1.5)),
                    period=Observation(distribution="uniform", bounds=(9.0, 11.0)),
                    eccentricity=Observation(distribution="uniform", bounds=(0.0, 0.3)),
                ),
                PlanetObservations(
                    name="c",
                    minimum_mass=Observation(mean=2.0, error=0.2),
                    period=Observation(mean=20.0, error=1.0),
                    eccentricity=Observation(mean=0.2, error=0.04),
                ),
            ],
        )

        theta_in = np.array(
            [
                45.0,
                1.0,
                1.0,
                2.0,
                10.0,
                20.0,
                0.1,
                0.2,
                180.0,
                200.0,
            ]
        )
        theta_out = theta_in.copy()
        theta_out[2] = 2.0

        assert np.isfinite(priors.log_prior(theta_in, system_obs)[0])
        assert priors.log_prior(theta_out, system_obs)[0] == -np.inf

    def test_log_prior_mixed_fixture_in_range(self, mixed_system_observations):
        """Mixed gaussian/uniform fixture should give finite prior in range."""
        theta = np.array(
            [
                40.0,
                0.9,
                1.0,
                2.0,
                10.0,
                20.0,
                0.15,
                0.2,
                120.0,
                240.0,
            ]
        )

        log_p = priors.log_prior(theta, mixed_system_observations)

        assert np.shape(log_p) == (1,)
        assert np.isfinite(log_p[0])

    def test_log_prior_mixed_fixture_rejects_uniform_period_out_of_range(
        self,
        mixed_system_observations,
    ):
        """Uniform bounds on period for planet b should be enforced."""
        theta = np.array(
            [
                40.0,
                0.9,
                1.0,
                2.0,
                12.5,
                20.0,
                0.15,
                0.2,
                120.0,
                240.0,
            ]
        )

        log_p = priors.log_prior(theta, mixed_system_observations)

        assert log_p[0] == -np.inf

    def test_log_prior_mixed_fixture_rejects_uniform_star_mass_out_of_range(
        self,
        mixed_uniform_star_mass_system_observations,
    ):
        """Uniform bounds on stellar mass should be enforced."""
        theta = np.array(
            [
                40.0,
                1.3,
                0.5,
                1.6,
                10.0,
                20.0,
                0.1,
                0.15,
                100.0,
                200.0,
            ]
        )

        log_p = priors.log_prior(theta, mixed_uniform_star_mass_system_observations)

        assert log_p[0] == -np.inf

    def test_log_prior_mixed_fixture_batch_rejects_only_invalid_rows(
        self,
        mixed_system_observations,
    ):
        """2D theta should preserve row-wise finite/-inf behavior."""
        theta = np.array(
            [
                [0.9, 40.0, 1.0, 2.0, 10.0, 20.0, 0.10, 0.2, 100.0, 200.0],
                [0.9, 40.0, 2.0, 2.0, 10.0, 20.0, 0.10, 0.2, 100.0, 200.0],
                [0.9, 40.0, 1.0, 2.0, 10.0, 20.0, 0.10, 0.2, 100.0, 200.0],
            ]
        )
        theta[2, 0] = 0.7  # Still finite under Gaussian star-mass prior.

        log_p = priors.log_prior(theta, mixed_system_observations)

        assert log_p.shape == (3,)
        assert np.isfinite(log_p[0])
        assert log_p[1] == -np.inf
        assert np.isfinite(log_p[2])

    def test_log_prior_rejects_eccentricity_above_one(
        self,
        simple_system_observations,
    ):
        """Eccentricity terms should be truncated above 1.0."""
        theta = np.array(
            [
                1.0,
                45.0,
                1.0,
                2.0,
                10.0,
                20.0,
                1.1,
                0.2,
                180.0,
                200.0,
            ]
        )

        log_p = priors.log_prior(theta, simple_system_observations)

        assert log_p[0] == -np.inf

    def test_log_inclination_prior_vector_handles_out_of_range(self):
        """Vector inputs with out-of-range values should be set to -inf."""
        vals = np.array([0.0, 45.0, 200.0])
        lp = priors.log_inclination_prior(vals)

        assert lp.shape == (3,)
        assert lp[0] == -np.inf
        assert lp[2] == -np.inf
        # middle value should be finite and match expected log(sin(i)) behaviour
        assert lp[1] == approx(np.log(np.sin(np.radians(45.0))))
