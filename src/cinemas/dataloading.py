"""
dataloading.py
==============
Functions for loading and processing observational data for the CINEMAS analysis.
"""

import numpy as np
import pandas as pd

from . import constants
from . import observation_classes as obs


def select_compact_multiplanet_rv_systems(
    exoplanet_catalogue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects multi-planet systems discovered by the radial velocity method from an
    exoplanet.eu-like catalogue
    """
    compact_multiplanet_rv_systems = exoplanet_catalogue.groupby("star_name").filter(
        lambda x: (x["detection_type"] == "Radial Velocity").all()
        & (len(x) > 2)
        & is_compact(x)
    )

    return compact_multiplanet_rv_systems


def is_compact(system_data: pd.DataFrame) -> bool:
    """
    Checks if there is at least one triplet of planets with period ratios between
    adjacent planets < 2 (see Tamayo+20).
    """
    system_data = system_data.sort_values("orbital_period")
    periods = system_data["orbital_period"].values
    period_ratios = periods[1:] / periods[:-1]
    return np.any((period_ratios[:-1] < 2) & (period_ratios[1:] < 2))


def get_system_data(star_name: str, catalogue: pd.DataFrame) -> pd.DataFrame:
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
