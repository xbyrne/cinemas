"""
test_mcmc.py
============
Tests for MCMC helper functions in the CINEMAS package.
"""

import numpy as np
import pytest
from pytest import approx

from cinemas import mcmc


class TestLogPosterior:
    """Tests for posterior evaluation logic."""

    def test_log_posterior_1d_returns_minus_inf_for_unphysical_prior(
        self, simple_system_observations, monkeypatch
    ):
        """Likelihood should not be evaluated if prior is non-finite."""

        def fake_log_prior(theta, system_obs):
            return -np.inf

        def fake_log_likelihood(theta, spock_classifier):
            raise AssertionError("Likelihood should not be called for -inf prior")

        monkeypatch.setattr(mcmc.priors, "log_prior", fake_log_prior)
        monkeypatch.setattr(mcmc.likelihood, "log_likelihood", fake_log_likelihood)

        theta = np.array([45.0, 1.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0])
        log_post = mcmc.log_posterior(theta, simple_system_observations)

        assert log_post == -np.inf

    def test_log_posterior_1d_adds_prior_and_likelihood(
        self, simple_system_observations, monkeypatch
    ):
        """Finite prior should be combined with likelihood for 1D theta."""
        monkeypatch.setattr(mcmc.priors, "log_prior", lambda theta, system_obs: -2.0)
        monkeypatch.setattr(
            mcmc.likelihood,
            "log_likelihood",
            lambda theta, spock_classifier: -3.5,
        )

        theta = np.array([45.0, 1.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0])
        log_post = mcmc.log_posterior(theta, simple_system_observations)

        assert log_post == approx(-5.5)

    def test_log_posterior_2d_evaluates_only_physical_samples(
        self, simple_system_observations, monkeypatch
    ):
        """Only finite-prior rows should be sent to the likelihood."""
        captured = {}

        def fake_log_prior(theta, system_obs):
            return np.array([1.0, -np.inf, 3.0])

        def fake_log_likelihood(theta, spock_classifier):
            captured["theta"] = theta
            return np.array([10.0, 20.0])

        monkeypatch.setattr(mcmc.priors, "log_prior", fake_log_prior)
        monkeypatch.setattr(mcmc.likelihood, "log_likelihood", fake_log_likelihood)

        theta = np.array(
            [
                [45.0, 1.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0],
                [0.01, 1.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0],
                [50.0, 1.2, 1.1, 2.1, 9.0, 19.0, 0.12, 0.22, 170.0, 210.0],
            ]
        )

        log_post = mcmc.log_posterior(theta, simple_system_observations)

        assert captured["theta"].shape[0] == 2
        assert np.allclose(log_post, [11.0, -np.inf, 23.0])

    def test_log_posterior_2d_skips_likelihood_when_no_samples_are_physical(
        self, simple_system_observations, monkeypatch
    ):
        """An all-unphysical batch should not call the likelihood at all."""

        def fake_log_prior(theta, system_obs):
            return np.array([-np.inf, -np.inf])

        def fake_log_likelihood(theta, spock_classifier):
            raise AssertionError("Likelihood should not be called for empty batch")

        monkeypatch.setattr(mcmc.priors, "log_prior", fake_log_prior)
        monkeypatch.setattr(mcmc.likelihood, "log_likelihood", fake_log_likelihood)

        theta = np.array(
            [
                [45.0, 1.0, 1.0, 2.0, 10.0, 20.0, 0.1, 0.2, 180.0, 200.0],
                [50.0, 1.2, 1.1, 2.1, 9.0, 19.0, 0.12, 0.22, 170.0, 210.0],
            ]
        )

        log_post = mcmc.log_posterior(theta, simple_system_observations)

        assert np.all(np.isneginf(log_post))

    def test_log_posterior_raises_for_invalid_theta_dim(
        self, simple_system_observations
    ):
        """Theta with ndim not in [1, 2] should raise ValueError."""
        theta = np.zeros((2, 2, 2))

        with pytest.raises(AssertionError, match="1D or 2D"):
            mcmc.log_posterior(theta, simple_system_observations)


class TestProposeTheta:
    """Tests for initial-theta proposal helper."""

    def test_propose_theta_shape_and_ranges(self, simple_system_observations):
        """Proposed theta should have expected length and physical ranges."""
        theta = mcmc.propose_theta(simple_system_observations)

        n_planets = simple_system_observations.n_planets
        n_parameters = 2 + 4 * n_planets

        assert theta.shape == (n_parameters,)
        assert 30.0 <= theta[0] <= 40.0
        assert theta[1] > 0.0

        minimum_masses = theta[2 : 2 + n_planets]
        periods = theta[2 + n_planets : 2 + 2 * n_planets]
        eccentricities = theta[2 + 2 * n_planets : 2 + 3 * n_planets]
        omegas = theta[2 + 3 * n_planets : 2 + 4 * n_planets]

        assert np.all(minimum_masses > 0.0)
        assert np.all(periods > 0.0)
        assert np.all((eccentricities >= 0.0) & (eccentricities <= 1.0))
        assert np.all((omegas >= 175.0) & (omegas <= 185.0))


class TestGenerateInitialStates:
    """Tests for generating valid initial MCMC walker states."""

    def test_generate_initial_states_collects_finite_walkers(
        self, simple_system_observations, monkeypatch
    ):
        """Function should keep sampling until enough finite-posterior states exist."""

        class DummyClassifier:
            pass

        class DummyTqdm:
            def __init__(self, total, desc, unit):
                self.total = total
                self.desc = desc
                self.unit = unit
                self.updated = 0
                self.postfix = None
                self.closed = False

            def update(self, n):
                self.updated += n

            def set_postfix(self, data):
                self.postfix = data
                return None

            def close(self):
                self.closed = True

        proposed = [
            np.full(10, 1.0),
            np.full(10, 2.0),
            np.full(10, 3.0),
        ]
        proposal_iter = iter(proposed)

        monkeypatch.setattr(mcmc, "FeatureClassifier", DummyClassifier)
        monkeypatch.setattr(mcmc, "tqdm", DummyTqdm)
        monkeypatch.setattr(
            mcmc,
            "propose_theta",
            lambda system_obs: next(proposal_iter),
        )

        def fake_log_posterior(theta, system_obs, spock_classifier):
            if np.all(theta == 2.0):
                return -np.inf
            return 0.0

        monkeypatch.setattr(mcmc, "log_posterior", fake_log_posterior)

        states = mcmc.generate_initial_states(simple_system_observations, nwalkers=2)

        assert states.shape == (2, 10)
        assert np.all(states[0] == 1.0)
        assert np.all(states[1] == 3.0)

    def test_generate_initial_states_raises_if_not_enough_valid_states(
        self, simple_system_observations, monkeypatch
    ):
        """Function should raise if max_tries is exhausted without enough walkers."""

        class DummyClassifier:
            pass

        class DummyTqdm:
            def __init__(self, total, desc, unit):
                self.total = total
                self.desc = desc
                self.unit = unit
                self.updated = 0
                self.postfix = None
                self.closed = False

            def update(self, n):
                self.updated += n

            def set_postfix(self, data):
                self.postfix = data
                return None

            def close(self):
                self.closed = True

        monkeypatch.setattr(mcmc, "FeatureClassifier", DummyClassifier)
        monkeypatch.setattr(mcmc, "tqdm", DummyTqdm)
        monkeypatch.setattr(mcmc, "propose_theta", lambda system_obs: np.full(10, 1.0))
        monkeypatch.setattr(
            mcmc,
            "log_posterior",
            lambda theta, system_obs, spock_classifier: -np.inf,
        )

        with pytest.raises(
            ValueError,
            match="Could not generate enough initial states",
        ):
            mcmc.generate_initial_states(simple_system_observations, nwalkers=1)


class TestRunMCMCSampling:
    """Tests for MCMC sampling orchestration."""

    def test_run_mcmc_sampling_uses_provided_initial_states(
        self, simple_system_observations, monkeypatch
    ):
        """Sampler should run with provided initial states and return its outputs."""
        captured = {}

        class DummyClassifier:
            pass

        class DummySampler:
            def __init__(self, nwalkers, ndim, log_prob_fn, args, vectorize):
                captured["nwalkers"] = nwalkers
                captured["ndim"] = ndim
                captured["vectorize"] = vectorize
                captured["args"] = args
                captured["log_prob_fn"] = log_prob_fn
                self.acceptance_fraction = np.array([0.2, 0.4])

            def run_mcmc(self, initial_states, nsteps, progress):
                captured["initial_states"] = initial_states
                captured["nsteps"] = nsteps
                captured["progress"] = progress

            def get_chain(self):
                return np.zeros((5, 2, 10))

            def get_autocorr_time(self):
                return np.array([12.0, 13.0])

        monkeypatch.setattr(mcmc, "FeatureClassifier", DummyClassifier)
        monkeypatch.setattr(mcmc, "EnsembleSampler", DummySampler)

        initial_states = np.ones((2, 10))
        samples, tau, acceptance_fraction = mcmc.run_mcmc_sampling(
            simple_system_observations,
            nwalkers=2,
            nsteps=5,
            initial_states=initial_states,
        )

        assert captured["nwalkers"] == 2
        assert captured["ndim"] == 10
        assert captured["vectorize"] is True
        assert captured["log_prob_fn"] is mcmc.log_posterior
        assert isinstance(captured["args"][1], DummyClassifier)
        assert np.array_equal(captured["initial_states"], initial_states)
        assert captured["nsteps"] == 5
        assert captured["progress"] is True

        assert samples.shape == (5, 2, 10)
        assert np.allclose(tau, [12.0, 13.0])
        assert np.allclose(acceptance_fraction, [0.2, 0.4])

    def test_run_mcmc_sampling_uses_default_nwalkers_and_autocorr_fallback(
        self, simple_system_observations, monkeypatch
    ):
        """Autocorrelation failures should return tau = -1.0."""

        class DummyClassifier:
            pass

        class DummyAutocorrError(Exception):
            pass

        class DummySampler:
            def __init__(self, nwalkers, ndim, log_prob_fn, args, vectorize):
                self.acceptance_fraction = np.array([0.5] * nwalkers)

            def run_mcmc(self, initial_states, nsteps, progress):
                return None

            def get_chain(self):
                return np.zeros((3, 20, 10))

            def get_autocorr_time(self):
                raise DummyAutocorrError("too short")

        generated_states = np.ones((20, 10))

        monkeypatch.setattr(mcmc, "FeatureClassifier", DummyClassifier)
        monkeypatch.setattr(mcmc, "AutocorrError", DummyAutocorrError)
        monkeypatch.setattr(mcmc, "EnsembleSampler", DummySampler)
        monkeypatch.setattr(
            mcmc,
            "generate_initial_states",
            lambda system_obs, nwalkers: generated_states,
        )

        samples, tau, acceptance_fraction = mcmc.run_mcmc_sampling(
            simple_system_observations,
            nwalkers=None,
            nsteps=3,
            initial_states=None,
        )

        assert samples.shape == (3, 20, 10)
        assert tau == -1.0
        assert acceptance_fraction.shape == (20,)

    def test_log_posterior_raises_on_bad_theta_dim(self, simple_system_observations):
        """3D theta should raise AssertionError."""
        theta = np.zeros((2, 2, 2))

        with pytest.raises(AssertionError):
            mcmc.log_posterior(theta, simple_system_observations)

    def test_log_posterior_handles_unphysical_samples(
        self, monkeypatch, simple_system_observations
    ):
        """2D theta with one unphysical sample should skip likelihood for it."""
        n_planets = simple_system_observations.n_planets
        n_params = 2 + 4 * n_planets
        theta = np.zeros((2, n_params))

        # Monkeypatch priors.log_prior to return one -inf and one finite value
        monkeypatch.setattr(
            "cinemas.priors.log_prior",
            lambda th, so: np.array([-np.inf, 0.0])
        )

        # Monkeypatch likelihood.log_likelihood to be called for the single sample
        monkeypatch.setattr(
            "cinemas.likelihood.log_likelihood", lambda th, sp: np.array([0.123])
        )

        out = mcmc.log_posterior(theta, simple_system_observations)

        assert out.shape == (2,)
        assert out[0] == -np.inf
        assert out[1] == pytest.approx(0.123)
