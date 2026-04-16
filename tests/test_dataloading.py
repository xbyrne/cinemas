"""
test_dataloading.py
===================
Tests for the data loading functions in the CINEMAS package.
"""

from pytest import approx

from cinemas.dataloading import (
    is_compact,
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
