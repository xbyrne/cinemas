"""
dataloading.py
==============
Functions for loading and processing observational data for the CINEMAS analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyvo as vo

from . import constants
from . import observation_classes as obs

# ============================
# Downloading + selecting data


def download_multiplanet_systems(
    save_path: Path | None = constants.DEFAULT_DOWNLOAD_PATH,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download the exoplanet catalogue from the NASA Exoplanet Archive (just the multi-
    planet systems), save it as a CSV file (if needed), and return it as a DataFrame.
    """

    if save_path is not None and save_path.exists() and not overwrite:
        print(f"Input catalogue already exists at {save_path}; loading from there.")
        print("Set `overwrite=True` to re-download the catalogue.")
        return pd.read_csv(save_path)

    service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")

    query = (
        "SELECT "
        + ", ".join(constants.ALL_FIELDS)
        + " FROM pscomppars WHERE sy_pnum > 2"
    )

    result = service.search(query)
    catalogue = result.to_table().to_pandas()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue.to_csv(save_path, index=False)

    return catalogue


def select_compact_multiplanet_rv_systems(
    exoplanet_catalogue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects multi-planet systems discovered by the radial velocity method from a NASA
    Exoplanet Archive-style catalogue.
    """
    compact_multiplanet_rv_systems = exoplanet_catalogue.groupby("hostname").filter(
        lambda x: (x["discoverymethod"] == "Radial Velocity").all()
        & (len(x) > 2)
        & is_compact(x)
        & (x["pl_controv_flag"] == 0).all()
    )

    return compact_multiplanet_rv_systems


def is_compact(system_data: pd.DataFrame) -> bool:
    """
    Checks if there is at least one triplet of planets with period ratios between
    adjacent planets < 2 (see Tamayo+20).
    """
    system_data = system_data.sort_values("pl_orbper")
    periods = system_data["pl_orbper"].values
    period_ratios = periods[1:] / periods[:-1]
    return np.any((period_ratios[:-1] < 2) & (period_ratios[1:] < 2))


# ===============
# Organising data


def get_system_data(star_name: str, catalogue: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the relevant data for a given star from the catalogue, to be used as priors.
    NB: Masses are returned in M_E and periods in days.
    """

    system_catalogue = catalogue[catalogue["hostname"] == star_name]
    system_data = system_catalogue[constants.ALL_FIELDS].copy()

    return system_data


def load_system_observations(
    star_name: str, exoplanet_catalogue: pd.DataFrame
) -> obs.SystemObservations:
    """
    Load the prior constraints for a given star from the catalogue, and return them as a
    SystemObservations object.
    """
    compact_multiplanet_rv_systems = select_compact_multiplanet_rv_systems(
        exoplanet_catalogue
    )
    system_data = get_system_data(star_name, compact_multiplanet_rv_systems)

    planet_observations = []
    for planet in system_data["pl_name"]:
        planet_data_row = system_data[system_data["pl_name"] == planet].iloc[0]
        planet_obs = package_planet_observations(planet_data_row)
        planet_observations.append(planet_obs)

    star_mass_mean = system_data["st_mass"].iloc[0]
    star_mass_error = get_average_param_error(system_data.iloc[0], "st_mass")
    star_mass_obs = obs.Observation(
        distribution="gaussian", mean=star_mass_mean, error=star_mass_error
    )

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
    eccentricity = planet_data_row["pl_orbeccen"]
    if pd.isna(eccentricity) or eccentricity == 0.0:
        eccentricity_obs = obs.Observation(
            distribution="gaussian",
            mean=0.05,
            error=0.05,
        )
    else:
        eccentricity_error = get_average_param_error(planet_data_row, "pl_orbeccen")
        eccentricity_obs = obs.Observation(
            distribution="gaussian", mean=eccentricity, error=eccentricity_error
        )

    minimum_mass_obs = obs.Observation(
        distribution="gaussian",
        mean=planet_data_row["pl_msinie"],
        error=get_average_param_error(planet_data_row, "pl_msinie"),
    )
    period_obs = obs.Observation(
        distribution="gaussian",
        mean=planet_data_row["pl_orbper"],
        error=get_average_param_error(planet_data_row, "pl_orbper"),
    )

    return obs.PlanetObservations(
        name=planet_data_row["pl_name"],
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
        np.abs(planet_data_row[f"{param_name}err1"])
        + np.abs(planet_data_row[f"{param_name}err2"])  # This one is -ve.
    )
