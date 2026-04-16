"""
test_dataloading.py
===================
Tests for the data loading functions in the CINEMAS package.
"""

from pytest import approx

from cinemas import package_planet_observations


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
