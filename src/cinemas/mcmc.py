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
from emcee.moves import DEMove, DESnookerMove
import numpy as np
from spock import FeatureClassifier
from tqdm import tqdm

from . import likelihood, observation_classes as obs, priors


# Module-level place to hold the SystemObservations and FeatureClassifiers.
# This allows `log_posterior` to access the classifier and data without them being
# passed as arguments to `EnsembleSampler.log_prob_fn`, which avoids costly
# argument pickling when using a multiprocessing pool.

spock_classifier = FeatureClassifier()  # Load the SPOCK classifier once at module level
current_system_obs: obs.SystemObservations | None = None


## ========
## Run MCMC


def run_mcmc_sampling(
    system_obs: obs.SystemObservations,
    nsteps: int = 10000,
    nwalkers: int = None,
    moves: list[tuple] = None,
    initial_states: np.ndarray = None,
) -> tuple[np.ndarray, float, float]:
    """
    Run MCMC sampling to obtain posterior samples for the system parameters.
    If given, `initial_states` should be an array of shape (nwalkers, n_parameters).
    """
    # Multiprocessing configs
    multiprocessing.set_start_method("fork")
    os.environ["OMP_NUM_THREADS"] = "1"

    # Set the module-level `current_system_obs` so that `log_posterior`
    # can access it without requiring it to be passed to the sampler (and
    # therefore avoiding repeated pickling when using multiprocessing).
    global current_system_obs
    current_system_obs = system_obs

    n_planets = system_obs.n_planets

    if nwalkers is None:
        print("Number of walkers not specified. Using default of 3(1 + 4 n_planets),")
        # Factor of 3 is a trade-off between better sampling and longer runtime
        nwalkers = 3 * (1 + 4 * n_planets)
        print(f" which in this case is {nwalkers} walkers ({n_planets} planets).")

    if moves is None:
        # Optimised set of moves
        moves = [(DEMove(gamma0=0.2), 0.9), (DESnookerMove(), 0.1)]

    if initial_states is None:
        # Initialize walkers in a small Gaussian ball around the observed values
        initial_states = generate_initial_states(system_obs, nwalkers)

    try:
        with Pool() as pool:
            # ^ Multiprocessing turns out to be faster than vectorising
            # Effectively, parallelising seems to be quicker at the `emcee` level than
            # at the `SPOCK` level.
            sampler = EnsembleSampler(
                nwalkers=nwalkers,
                ndim=1 + 4 * system_obs.n_planets,
                log_prob_fn=log_posterior,
                pool=pool,
                moves=moves,
            )
            sampler.run_mcmc(initial_states, nsteps, progress=True)
    finally:
        # Clear the global to avoid retaining large observation objects.
        current_system_obs = None

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


# ==================
# Posterior function


def log_posterior(theta: np.ndarray) -> float | np.ndarray:
    """Compute log posterior for parameter vector `theta`.

    This function reads `current_system_obs` from module scope rather than
    receiving it as an argument. Callers must ensure `current_system_obs` is
    set (e.g. by `run_mcmc_sampling`) before invoking the sampler.
    """
    global current_system_obs

    if current_system_obs is None:
        raise RuntimeError(
            "current_system_obs is not set. Call run_mcmc_sampling which sets it."
        )

    system_obs = current_system_obs

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
    system_obs: obs.SystemObservations, nwalkers: int, max_tries: int = 1000
) -> np.ndarray:
    """
    Generate initial states for the MCMC walkers
    """
    global current_system_obs
    if current_system_obs is None:
        raise RuntimeError(
            "current_system_obs is not set. Call run_mcmc_sampling which sets it."
        )

    initial_states = []
    progress_bar = tqdm(
        total=nwalkers, desc="Generating initial states", unit=" walkers"
    )

    for attempt in range(max_tries):
        theta_0 = propose_theta(system_obs)
        lp = log_posterior(theta_0)

        if np.isfinite(lp):
            initial_states.append(theta_0)
            progress_bar.update(1)
            progress_bar.set_postfix({"Tries": f"{attempt + 1}/{max_tries}"})
            if len(initial_states) >= nwalkers:
                break

    progress_bar.close()
    if len(initial_states) < nwalkers:
        raise RuntimeError(
            f"Only generated {len(initial_states)} valid initial states"
            + f" in {max_tries} attempts."
        )
    print(
        f"Generated {len(initial_states)} valid initial states"
        + f" in {attempt + 1} attempts."
    )

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
    d_omegas = np.random.uniform(0, 360, size=system_obs.n_planets - 1)

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
