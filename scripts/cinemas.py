"""
cinemas.py
==========
Constraining INclinations of Exoplanets and their MAsses by Stability
---------------------------------------------------------------------
A framework for obtaining Bayesian constraints on the inclinations, and hence true
masses (as well as other orbital parameters) of exoplanets in multi-planet systems by
using the constraint of long-term dynamical stability. The framework uses the stability
classifier from SPOCK. The result? Absolute cinema.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError
import rebound
from spock import FeatureClassifier

import constants
import observation_classes as obs

# ============
# Data loading


def select_multiplanet_rv_systems(catalogue_path: Path) -> pd.DataFrame:
    """
    Selects multi-planet systems discovered by the radial velocity method from an
    exoplanet.eu-like catalogue
    """
    exoplanet_catalogue = pd.read_csv(catalogue_path)

    rv_exoplanets = exoplanet_catalogue[
        exoplanet_catalogue["detection_type"] == "Radial Velocity"
    ]

    multiplanet_rv_systems = rv_exoplanets.groupby("star_name").filter(
        lambda x: len(x) > 2
    )

    return multiplanet_rv_systems


def is_compact(system_data: pd.DataFrame) -> bool:
    """
    Checks if there is at least one triplet of planets with period ratios between
    adjacent planets < 2 (see Tamayo+20).
    """
    periods = system_data["orbital_period"].values
    period_ratios = periods[1:] / periods[:-1]
    compact_pairs = np.sum(period_ratios < 2)
    return compact_pairs >= 2


def get_system_data(catalogue: pd.DataFrame, star_name: str) -> pd.DataFrame:
    """
    Extract the relevant data for a given star from the catalogue, to be used as priors.
    NB: Masses are returned in M_E and periods in days.
    """
    columns_to_return = (
        ["name", "planet_status"]
        + constants.FIELDS_TO_USE
        + [f"{field}_error_min" for field in constants.FIELDS_TO_USE]
        + [f"{field}_error_max" for field in constants.FIELDS_TO_USE]
    )

    system_catalogue = catalogue[catalogue["star_name"] == star_name]
    system_data = system_catalogue[columns_to_return]

    # Convert masses to Earth masses (initially in Jupiter masses)
    system_data.loc[
        :, system_data.columns.str.startswith("mass")
    ] *= constants.MJUP_MEARTH

    return system_data


def load_system_observations(
    star_name: str, catalogue_path: Path
) -> obs.SystemObservations:
    """
    Load the prior constraints for a given star from the catalogue, and return them as a
    SystemObservations object.
    """
    multiplanet_rv_systems = select_multiplanet_rv_systems(catalogue_path)
    system_data = get_system_data(multiplanet_rv_systems, star_name)

    planet_observations = []
    for planet in system_data["name"]:
        planet_data_row = system_data[system_data["name"] == planet].iloc[0]
        planet_obs = package_planet_observations(planet_data_row)
        planet_observations.append(planet_obs)

    star_mass_mean = system_data["star_mass"].iloc[0]
    star_mass_error = 0.5 * (
        system_data["star_mass_error_min"].iloc[0]
        + system_data["star_mass_error_max"].iloc[0]
    )
    star_mass_obs = obs.Observation(mean=star_mass_mean, error=star_mass_error)

    system_obs = obs.SystemObservations(
        star_name=star_name,
        star_mass=star_mass_obs,
        planet_observations=planet_observations,
    )

    return system_obs


def package_planet_observations(planet_data_row: pd.Series) -> obs.PlanetObservations:
    """
    Package the observations for a single planet into a PlanetObservations object.
    """
    # Handle missing eccentricity
    eccentricity = planet_data_row["eccentricity"]
    if pd.isna(eccentricity) or eccentricity == 0.0:
        eccentricity_obs = obs.Observation(mean=0.05, error=0.05)
    else:
        eccentricity_error = get_average_param_error(planet_data_row, "eccentricity")
        eccentricity_obs = obs.Observation(mean=eccentricity, error=eccentricity_error)

    minimum_mass_obs = obs.Observation(
        mean=planet_data_row["mass_sini"],
        error=get_average_param_error(planet_data_row, "mass_sini"),
    )
    period_obs = obs.Observation(
        mean=planet_data_row["orbital_period"],
        error=get_average_param_error(planet_data_row, "orbital_period"),
    )

    return obs.PlanetObservations(
        name=planet_data_row["name"],
        minimum_mass=minimum_mass_obs,
        period=period_obs,
        eccentricity=eccentricity_obs,
    )


def get_average_param_error(planet_data_row: pd.Series, param_name: str) -> float:
    """
    Get the average error for a given parameter from the planet data row.
    We're not dealing with asymmetric errors here; just take the average.
    """
    return 0.5 * (
        planet_data_row[f"{param_name}_error_min"]
        + planet_data_row[f"{param_name}_error_max"]
    )


# ==================
# REBOUND simulation creators


def create_rebound_simulations(
    star_mass: float | np.ndarray,
    masses: np.ndarray,
    periods: np.ndarray,
    eccentricities: np.ndarray = None,
    omegas: np.ndarray = None,
) -> rebound.Simulation | list[rebound.Simulation]:
    """
    Create REBOUND simulations for a given set of orbital parameters.
    Masses should be in Earth masses, and periods in days.
    Eccentricities and omegas are optional, and will be set to 0 if not provided.
    """
    if eccentricities is None:
        eccentricities = np.zeros_like(masses)
    if omegas is None:
        omegas = np.zeros_like(masses)

    if (isinstance(star_mass, np.ndarray) and star_mass.ndim == 1) or isinstance(
        star_mass, list
    ):
        # Multiple simulations
        simulations = []

        for star_mass_val, mass, period, ecc, omega in zip(
            star_mass, masses, periods, eccentricities, omegas
        ):
            # Recursion ftw
            simulations.append(
                create_rebound_simulations(star_mass_val, mass, period, ecc, omega)
            )

        return simulations

    # Single simulation
    sim = rebound.Simulation()

    sim.add(m=star_mass)

    for mass, period, ecc, omega in zip(masses, periods, eccentricities, omegas):
        sim.add(m=mass / constants.MSUN_MEARTH, P=period, e=ecc, omega=omega)

    sim.move_to_com()
    return sim


def create_rebound_simulations_from_theta(
    theta: np.ndarray,
) -> rebound.Simulation | list[rebound.Simulation]:
    """
    Creates REBOUND simulation(s) from the parameter vector `theta`.
    The shape of `theta` should be either (n_parameters,) or (n_samples, n_parameters).
    """

    star_mass, inclination, minimum_masses, periods, eccentricities, omegas = (
        unpack_theta(theta)
    )

    # minimum_masses are either of shape (n_planets,) or (n_samples, n_planets);
    # inclination is either of shape (1,) or (n_samples,).
    # We want the true masses, which are minimum_masses / sin(inclination).
    # To do this, we need to ensure that the shapes are compatible for broadcasting.
    if inclination.ndim == 1 and minimum_masses.ndim == 2:
        inclination = inclination[:, None]

    masses = minimum_masses / np.sin(np.radians(inclination))

    return create_rebound_simulations(
        star_mass, masses, periods, eccentricities, omegas
    )


def unpack_theta(theta: np.ndarray):
    """
    Unpack the parameter vector `theta` into its components.
    `theta` should either be of shape (n_parameters,) or (n_samples, n_parameters),
    where n_parameters = 2 + 4 * n_planets (star mass, inclination, minimum masses,
    periods, eccentricities, omegas).
    """
    assert theta.ndim in [1, 2], "`theta` should be either 1D or 2D array"

    assert (
        theta.shape[-1] - 2
    ) % 4 == 0, "`theta` should have 2 + 4 * n_planets parameters"
    n_planets = (theta.shape[-1] - 2) // 4

    star_mass = theta[..., 0]
    inclination = theta[..., 1]
    minimum_masses = theta[..., 2 : 2 + n_planets]
    periods = theta[..., 2 + n_planets : 2 + 2 * n_planets]
    eccentricities = theta[..., 2 + 2 * n_planets : 2 + 3 * n_planets]
    omegas = theta[..., 2 + 3 * n_planets : 2 + 4 * n_planets]

    return star_mass, inclination, minimum_masses, periods, eccentricities, omegas


# ==================
# Bayesian functions

# ------
# Priors


def log_inclination_prior(
    inclination_deg: float | np.ndarray, i_min=constants.I_MIN, i_max=constants.I_MAX
) -> float | np.ndarray:
    """
    Log prior for the inclination of a planet, assuming isotropic orientations.
    """

    log_prior = np.log(np.sin(np.radians(np.clip(inclination_deg, i_min, i_max))))

    if log_prior.ndim == 0:
        if inclination_deg < i_min or inclination_deg > i_max:
            log_prior = -np.inf
    else:
        log_prior[(inclination_deg < i_min) | (inclination_deg > i_max)] = -np.inf

    return log_prior


def log_gaussian_prior(
    x: float | np.ndarray, mean: float | np.ndarray, error: float | np.ndarray
) -> float | np.ndarray:
    """
    Log prior for a Gaussian distribution; negative values are truncated to -inf.
    The shapes of the inputs should be either:
    - `x`: (1,), (n_planets,), or (n_samples, n_planets)
    - `mean`: (1,), (n_planets,), or (n_planets,)
    - `error`: (1,), (n_planets,), or (n_planets,)
    The output will have the same shape as `x`.
    """
    # Vectorising
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)
    error = np.atleast_1d(error)

    log_prior = -0.5 * ((x - mean) / error) ** 2 - np.log(error * np.sqrt(2 * np.pi))

    log_prior[x < 0] = -np.inf

    return log_prior


def log_uniform_prior(x: float | np.ndarray, x_min: float, x_max: float) -> float:
    """
    Log prior for a uniform distribution; values outside the range are truncated to -inf
    """
    log_prior = -np.log(x_max - x_min) * np.ones_like(x)

    log_prior[(x < x_min) | (x > x_max)] = -np.inf

    return log_prior


def log_prior(
    theta: np.ndarray, system_obs: obs.SystemObservations
) -> float | np.ndarray:
    """
    Log prior for the full parameter set.
    `theta` can be either a 1D array (single parameter set) or a 2D array (multiple
    parameter sets; shape (n_samples, n_parameters)).
    """
    star_mass, inclination, minimum_masses, periods, eccentricities, omegas = (
        unpack_theta(theta)
    )

    # Each of the following contributions to the log_prior should either be
    # - a scalar (if `theta` is 1D); or
    # - an array of shape (n_samples,) (if `theta` is 2D).

    # Star mass
    log_p = log_gaussian_prior(
        star_mass, system_obs.star_mass.mean, system_obs.star_mass.error
    )
    # Inclination
    log_p += log_inclination_prior(inclination)
    # Minimum masses
    log_p += log_gaussian_prior(
        minimum_masses, system_obs.minimum_masses, system_obs.minimum_masses_errors
    ).sum(axis=-1)
    # Periods
    log_p += log_gaussian_prior(
        periods, system_obs.periods, system_obs.periods_errors
    ).sum(axis=-1)
    # Eccentricities
    log_p += log_gaussian_prior(
        eccentricities,
        system_obs.eccentricities,
        system_obs.eccentricities_errors,
    ).sum(axis=-1)
    # Omegas (uniform between 0 and 360)
    log_p += log_uniform_prior(omegas, 0, 360).sum(axis=-1)

    return log_p


# ----------
# Likelihood


def log_likelihood(
    theta: np.ndarray, spock_classifier: FeatureClassifier = None
) -> float | np.ndarray:
    """
    Log likelihood for the stability of the system, as predicted by SPOCK.
    `theta` can be either a 1D array (single parameter set) or a 2D array (multiple
    parameter sets; shape (n_samples, n_parameters)).
    """

    if spock_classifier is None:
        print(
            "Warning: No SPOCK classifier provided; creating a new one."
            " This is inefficient; if you need to call this function multiple times,"
            " consider passing a single classifier."
        )
        spock_classifier = FeatureClassifier()

    sims = create_rebound_simulations_from_theta(theta)

    stability_prob = spock_classifier.predict_stable(sims)
    log_prob = np.log(stability_prob)

    return log_prob


# ---------
# Posterior


def log_posterior(
    theta: np.ndarray,
    system_obs: obs.SystemObservations,
    spock_classifier: FeatureClassifier = None,
) -> float | np.ndarray:

    log_p = log_prior(theta, system_obs)

    if theta.ndim == 1:  # and hence log_p is scalar
        if not np.isfinite(log_p):
            # Zero prior probability, so posterior is zero regardless of likelihood
            return -np.inf

        # Finite log_p, so we evaluate the likelihood
        log_l = log_likelihood(theta, spock_classifier)
        log_p += log_l

    elif theta.ndim == 2:  # and hence log_p is an array of shape (n_samples,)
        unphysical_samples = ~np.isfinite(log_p)
        log_l = log_likelihood(theta[~unphysical_samples], spock_classifier)
        log_p[~unphysical_samples] += log_l

    else:
        raise ValueError(
            f"`theta` should be either 1D or 2D array; got shape {theta.shape}"
        )

    return log_p


# =============
# MCMC sampling


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


# ==============================


def main():

    COMPACT_MULTIPLANET_RV_SYSTEMS = [
        "Barnard's star",
        "GJ 667 C",
        "HD 158259",
        "HD 184010",
        "HD 215152",
        "HD 28471",
        "HD 34445",
        "HD 38677",
        "HD 40307",
        "YZ Cet",
    ]

    EXOPLANET_CATALOGUE_PATH = Path(
        "../data/exoplanet.eu_catalog_15-03-26_22_54_01.csv"
    )
    RESULTS_DIR = Path("../results/mcmc")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for system in COMPACT_MULTIPLANET_RV_SYSTEMS:
        print("\n" + "=" * 50)
        print(f"In CINEMAS now: {system}...")

        results_path = RESULTS_DIR / f"{system.lower().replace(' ', '_')}_10k.npz"

        np.random.seed(42)

        system_obs = load_system_observations(system, EXOPLANET_CATALOGUE_PATH)

        samples, tau, acceptance_fraction = run_mcmc_sampling(
            system_obs, nsteps=10000, nwalkers=None  # Use default number of walkers
        )
        np.savez_compressed(
            results_path,
            samples=samples,
            tau=tau,
            acceptance_fraction=acceptance_fraction,
        )
        print(f"Results saved to {results_path.resolve()} .")
        print(f"Autocorrelation time: {tau}")
        print(f"Acceptance fraction: {acceptance_fraction}")

        print("\n" + "=" * 50 + "\n\n")


if __name__ == "__main__":
    main()
