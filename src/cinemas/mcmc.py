"""
mcmc.py
=======
Functions for running MCMC sampling for the CINEMAS package, using the emcee library.
"""

from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError
import numpy as np
from spock import FeatureClassifier
from tqdm import tqdm

from . import likelihood, observation_classes as obs, priors


# ==================
# Posterior function


def log_posterior(
    theta: np.ndarray,
    system_obs: obs.SystemObservations,
    spock_classifier: FeatureClassifier = None,
) -> float | np.ndarray:

    log_p = priors.log_prior(theta, system_obs)

    if theta.ndim == 1:  # and hence log_p is scalar
        if not np.isfinite(log_p):
            # Zero prior probability, so posterior is zero regardless of likelihood
            return -np.inf

        # Finite log_p, so we evaluate the likelihood
        log_l = likelihood.log_likelihood(theta, spock_classifier)
        log_p += log_l

    elif theta.ndim == 2:  # and hence log_p is an array of shape (n_samples,)
        unphysical_samples = ~np.isfinite(log_p)
        log_l = likelihood.log_likelihood(theta[~unphysical_samples], spock_classifier)
        log_p[~unphysical_samples] += log_l

    else:
        raise ValueError(
            f"`theta` should be either 1D or 2D array; got shape {theta.shape}"
        )

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
    max_tries = 1000
    progress_bar = tqdm(
        range(max_tries), total=max_tries, desc="Generating initial states"
    )
    spock_classifier = FeatureClassifier()
    for _ in progress_bar:
        theta_0 = propose_theta(system_obs)
        lp = log_posterior(theta_0, system_obs, spock_classifier)

        if np.isfinite(lp):
            initial_states.append(theta_0)
            progress_bar.set_postfix(
                {"Walkers found": f"{len(initial_states)}/{nwalkers}"}
            )
        if len(initial_states) >= nwalkers:
            break

    if len(initial_states) < nwalkers:
        raise ValueError(
            f"Could not generate enough initial states within {max_tries} tries."
            f" Found {len(initial_states)}/{nwalkers} valid initial states."
            f" Consider increasing `max_tries` or relaxing the priors."
        )

    print(f"Found {nwalkers} valid initial states in {_ + 1} tries.")
    initial_states = np.array(initial_states)
    return initial_states


def propose_theta(system_obs: obs.SystemObservations) -> np.ndarray:
    """
    Propose a single parameter vector `theta`, by sampling from the priors.
    This is used to generate initial states for the MCMC walkers.
    """
    stellar_mass = np.clip(
        np.random.normal(system_obs.star_mass.mean, system_obs.star_mass.error / 10),
        a_min=0.01,
        a_max=None,
    )
    inclination_deg = np.random.uniform(30, 40)
    minimum_masses = np.clip(
        np.random.normal(
            system_obs.minimum_masses, system_obs.minimum_masses_errors / 10
        ),
        a_min=0.01,
        a_max=None,
    )
    periods = np.clip(
        np.random.normal(system_obs.periods, system_obs.periods_errors / 10),
        a_min=0.001,
        a_max=None,
    )
    eccentricities = np.random.uniform(0, 1e-3, size=system_obs.n_planets)
    omegas = np.random.uniform(175, 185, size=system_obs.n_planets)

    proposed_theta = np.concatenate(
        (
            [stellar_mass, inclination_deg],
            minimum_masses,
            periods,
            eccentricities,
            omegas,
        )
    )
    return proposed_theta


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
        nwalkers = 2 * (2 + 4 * n_planets)

    if initial_states is None:
        # Initialize walkers in a small Gaussian ball around the observed values
        initial_states = generate_initial_states(system_obs, nwalkers)

    spock_classifier = FeatureClassifier()

    sampler = EnsembleSampler(
        nwalkers=nwalkers,
        ndim=2 + 4 * system_obs.n_planets,
        log_prob_fn=log_posterior,
        args=[system_obs, spock_classifier],
        vectorize=True,
    )
    sampler.run_mcmc(initial_states, nsteps, progress=True)

    samples = sampler.get_chain()
    acceptance_fraction = sampler.acceptance_fraction
    try:
        tau = sampler.get_autocorr_time()
    except AutocorrError as e:
        print(
            "Warning: Autocorrelation time could not be estimated reliably. Error:",
            e,
        )
        tau = -1.0

    return samples, tau, acceptance_fraction
