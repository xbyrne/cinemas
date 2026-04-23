"""
test_dataloading.py
===================
Tests for the data loading functions in the CINEMAS package.
"""

import pandas as pd
from pytest import approx

from cinemas import constants
from cinemas.dataloading import (
    get_average_param_error,
    get_system_data,
    is_compact,
    load_system_observations,
    package_planet_observations,
    select_compact_multiplanet_rv_systems,
)


def test_is_compact_on_compact_system(example_system_data_compact):
    """Test that is_compact correctly identifies a compact system."""
    assert is_compact(example_system_data_compact)


def test_is_compact_on_non_compact_system(example_system_data_non_compact):
    """Test that is_compact correctly identifies a non-compact system."""
    assert not is_compact(example_system_data_non_compact)


def test_select_compact_multiplanet_rv_systems(example_exoplanet_catalogue):
    """
    Test that select_compact_multiplanet_rv_systems correctly selects compact systems.
    """
    compact_systems = select_compact_multiplanet_rv_systems(example_exoplanet_catalogue)

    assert len(compact_systems) == 3
    assert set(compact_systems["star_name"].unique()) == {"Star A"}


def test_package_planet_observations_on_example_row(example_planet_data_row):
    """Test that package_planet_observations correctly packages a row of planet data."""
    planet_obs = package_planet_observations(example_planet_data_row)

    assert planet_obs.name == "Raxacoricofallapatorius"
    assert planet_obs.minimum_mass.mean == approx(1.0)
    assert planet_obs.minimum_mass.error == approx(0.15)
    assert planet_obs.period.mean == approx(10.0)
    assert planet_obs.period.error == approx(0.75)
    assert planet_obs.eccentricity.mean == approx(0.1)
    assert planet_obs.eccentricity.error == approx(0.03)


def test_package_planet_observations_on_example_row_without_eccentricity(
    example_planet_data_row_without_eccentricity,
):
    """Test that package_planet_observations correctly handles missing eccentricity."""
    planet_obs = package_planet_observations(
        example_planet_data_row_without_eccentricity
    )

    assert planet_obs.name == "Raxacoricovarlonpatorius"
    assert planet_obs.minimum_mass.mean == approx(1.0)
    assert planet_obs.minimum_mass.error == approx(0.15)
    assert planet_obs.period.mean == approx(10.0)
    assert planet_obs.period.error == approx(0.75)
    assert planet_obs.eccentricity.mean == approx(0.05)
    assert planet_obs.eccentricity.error == approx(0.05)


def test_get_system_data_filters_columns_and_converts_mass_units(
    example_catalogue_for_system_data,
):
    """Test get_system_data returns expected columns and converts mass units."""
    system_data = get_system_data("Star A", example_catalogue_for_system_data)

    assert list(system_data["name"]) == ["Planet b"]
    assert "planet_status" in system_data.columns

    # Columns beginning with "mass" are converted from Mjup to Mearth.
    assert system_data["mass"].iloc[0] == approx(constants.MJUP_MEARTH)
    assert system_data["mass_sini"].iloc[0] == approx(constants.MJUP_MEARTH)
    assert system_data["mass_error_min"].iloc[0] == approx(0.1 * constants.MJUP_MEARTH)
    assert system_data["mass_error_max"].iloc[0] == approx(0.2 * constants.MJUP_MEARTH)

    # "star_mass" should not be converted by the startswith("mass") rule.
    assert system_data["star_mass"].iloc[0] == approx(1.2)


def test_get_average_param_error_returns_arithmetic_mean(example_planet_data_row):
    """Test averaging of min/max parameter errors."""
    assert get_average_param_error(example_planet_data_row, "mass_sini") == approx(0.15)
    assert get_average_param_error(
        example_planet_data_row,
        "orbital_period",
    ) == approx(0.75)


def test_load_system_observations_builds_system_observations(monkeypatch):
    """Test load_system_observations packages planets and stellar constraints."""
    exoplanet_catalogue = pd.DataFrame(
        {
            "star_name": ["ignored"],
            "name": ["ignored"],
        }
    )

    system_data = pd.DataFrame(
        {
            "name": ["b", "c"],
            "star_mass": [1.0, 1.0],
            "star_mass_error_min": [0.04, 0.04],
            "star_mass_error_max": [0.06, 0.06],
            "mass_sini": [1.0, 2.0],
            "mass_sini_error_min": [0.1, 0.2],
            "mass_sini_error_max": [0.1, 0.2],
            "orbital_period": [10.0, 20.0],
            "orbital_period_error_min": [0.5, 1.0],
            "orbital_period_error_max": [0.5, 1.0],
            "eccentricity": [0.1, 0.2],
            "eccentricity_error_min": [0.02, 0.04],
            "eccentricity_error_max": [0.02, 0.04],
        }
    )

    monkeypatch.setattr(
        "cinemas.dataloading.select_compact_multiplanet_rv_systems",
        lambda catalogue: catalogue,
    )
    monkeypatch.setattr(
        "cinemas.dataloading.get_system_data",
        lambda star_name, catalogue: system_data,
    )

    system_obs = load_system_observations("Star A", exoplanet_catalogue)

    assert system_obs.star_name == "Star A"
    assert system_obs.star_mass.mean == approx(1.0)
    assert system_obs.star_mass.error == approx(0.05)

    assert system_obs.n_planets == 2
    assert [planet.name for planet in system_obs.planet_observations] == ["b", "c"]
    assert [obs.mean for obs in system_obs.minimum_masses] == [1.0, 2.0]
    assert [obs.mean for obs in system_obs.periods] == [10.0, 20.0]
    assert [obs.mean for obs in system_obs.eccentricities] == [0.1, 0.2]
