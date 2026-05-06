"""
mcmc.py
=======
Functions for running MCMC sampling for the CINEMAS package, using the emcee library.
"""

import multiprocessing
from multiprocessing import Pool
import os

from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError
import numpy as np
from spock import FeatureClassifier
from tqdm import tqdm

from . import likelihood, observation_classes as obs, priors

multiprocessing.set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"  # Avoid OpenMP issues with multiprocessing


# ==================
# Posterior function

spock_classifier = FeatureClassifier()  # Load the SPOCK classifier once at module level


def log_posterior(
    theta: np.ndarray, system_obs: obs.SystemObservations
) -> float | np.ndarray:

    log_p = priors.log_prior(theta, system_obs)

    if not np.isfinite(log_p):
        # Zero prior probability, so posterior is zero regardless of likelihood
        return -np.inf

    # Finite log_p, so we evaluate the likelihood
    log_l = likelihood.log_likelihood(theta, spock_classifier)
    log_p += log_l

    return log_p


# =========================
# Generating initial states


def generate_initial_states(
    system_obs: obs.SystemObservations, nwalkers: int
) -> np.ndarray:
    """
    Generate initial states for the MCMC walkers
    """
    initial_states = []
    progress_bar = tqdm(
        total=nwalkers, desc="Generating initial states", unit=" walkers"
    )

    for walker in tqdm(range(nwalkers)):
        theta_0 = propose_theta(system_obs)

        initial_states.append(theta_0)
        progress_bar.update(1)

        progress_bar.set_postfix({"Tries": f"{walker + 1}/{nwalkers}"})
        if len(initial_states) >= nwalkers:
            break

    progress_bar.close()

    initial_states = np.stack(initial_states)
    return initial_states


def propose_theta(system_obs: obs.SystemObservations) -> np.ndarray:
    """
    Propose a single parameter vector `theta`, by sampling from the priors.
    This is used to generate initial states for the MCMC walkers.
    """
    inclination_deg = np.random.uniform(55, 60)
    stellar_mass = np.clip(
        _propose_from_observation(system_obs.star_mass),
        a_min=0.01,
        a_max=None,
    )
    minimum_masses = np.clip(
        np.array(
            [
                _propose_from_observation(planet.minimum_mass)
                for planet in system_obs.planet_observations
            ]
        ),
        a_min=0.01,
        a_max=None,
    )
    periods = np.clip(
        np.array(
            [
                _propose_from_observation(planet.period)
                for planet in system_obs.planet_observations
            ]
        ),
        a_min=0.001,
        a_max=None,
    )
    eccentricities = np.random.uniform(0, 1e-2, size=system_obs.n_planets)
    d_omegas = np.random.uniform(175, 185, size=system_obs.n_planets - 1)

    proposed_theta = np.concatenate(
        (
            [inclination_deg, stellar_mass],
            minimum_masses,
            periods,
            eccentricities,
            d_omegas,
        )
    )
    return proposed_theta


def _propose_from_observation(observation: obs.Observation) -> float:
    """Draw one random value from an Observation prior."""
    if observation.distribution == "gaussian":
        return np.random.normal(observation.mean, observation.error / 10)

    if observation.distribution == "uniform":
        return np.random.uniform(observation.bounds[0], observation.bounds[1])

    raise ValueError(
        f"Unsupported observation distribution: {observation.distribution}"
    )


# =========================


def run_mcmc_sampling(
    system_obs: obs.SystemObservations,
    nwalkers: int = None,
    nsteps: int = 1000,
    initial_states: np.ndarray = None,
) -> tuple[np.ndarray, float, float]:
    """
    Run MCMC sampling to obtain posterior samples for the system parameters.
    If given, `initial_states` should be an array of shape (nwalkers, n_parameters).
    """

    n_planets = system_obs.n_planets

    if nwalkers is None:
        print("Number of walkers not specified. Using default of 2(1 + 4 n_planets),")
        nwalkers = 2 * (1 + 4 * n_planets)
        print(f" which in this case is {nwalkers} walkers ({n_planets} planets).")

    if initial_states is None:
        # Initialize walkers in a small Gaussian ball around the observed values
        initial_states = generate_initial_states(system_obs, nwalkers)

    with Pool() as pool:
        sampler = EnsembleSampler(
            nwalkers=nwalkers,
            ndim=1 + 4 * system_obs.n_planets,
            log_prob_fn=log_posterior,
            args=(system_obs,),
            pool=pool,
        )
        sampler.run_mcmc(initial_states, nsteps, progress=True)

    samples = sampler.get_chain()
    log_probs = sampler.get_log_prob()
    acceptance_fraction = sampler.acceptance_fraction
    try:
        tau = sampler.get_autocorr_time()
    except AutocorrError as e:
        print(
            "Warning: Autocorrelation time could not be estimated reliably. Error:",
            e,
        )
        tau = -1.0

    return samples, log_probs, tau, acceptance_fraction
