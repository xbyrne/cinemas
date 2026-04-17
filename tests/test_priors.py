"""
test_priors.py
==============
Tests for prior probability functions in the CINEMAS package.
"""

import numpy as np
from cinemas import priors
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


class TestFullPrior:
    """Tests for the full prior over theta."""

    def test_log_prior_on_single_theta_returns_finite(self, simple_system_observations):
        """A physically plausible theta should yield finite total log prior."""
        theta = np.array(
            [
                1.0,
                45.0,
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
                [1.0, 45.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0],
                [1.1, 50.0, 1.2, 2.1, 10.1, 19.9, 0.15, 0.25, 100.0, 250.0],
            ]
        )

        log_p = priors.log_prior(theta, simple_system_observations)

        assert log_p.shape == (2,)
        assert np.all(np.isfinite(log_p))

    def test_log_prior_rejects_unphysical_parameters(self, simple_system_observations):
        """Out-of-range inclination should produce -inf prior."""
        theta = np.array(
            [
                1.0,
                0.01,
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
