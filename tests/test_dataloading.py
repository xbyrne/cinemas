"""
test_dataloading.py
===================
Tests for the data loading functions in the CINEMAS package.
"""

import pandas as pd
from pytest import approx

from cinemas.dataloading import (
    download_multiplanet_systems,
    get_average_param_error,
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
    assert set(compact_systems["hostname"].unique()) == {"Star A"}


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


def test_get_average_param_error_returns_arithmetic_mean(example_planet_data_row):
    """Test averaging of min/max parameter errors."""
    assert get_average_param_error(example_planet_data_row, "pl_msinie") == approx(0.15)
    assert get_average_param_error(
        example_planet_data_row,
        "pl_orbper",
    ) == approx(0.75)


def test_load_system_observations_builds_system_observations(monkeypatch):
    """Test load_system_observations packages planets and stellar constraints."""
    exoplanet_catalogue = pd.DataFrame(
        {
            "hostname": ["ignored"],
            "pl_name": ["ignored"],
        }
    )

    system_data = pd.DataFrame(
        {
            "pl_name": ["b", "c"],
            "st_mass": [1.0, 1.0],
            "st_masserr1": [0.04, 0.04],
            "st_masserr2": [-0.06, -0.06],
            "pl_msinie": [1.0, 2.0],
            "pl_msinieerr1": [0.1, 0.2],
            "pl_msinieerr2": [-0.1, -0.2],
            "pl_orbper": [10.0, 20.0],
            "pl_orbpererr1": [0.5, 1.0],
            "pl_orbpererr2": [-0.5, -1.0],
            "pl_orbeccen": [0.1, 0.2],
            "pl_orbeccenerr1": [0.02, 0.04],
            "pl_orbeccenerr2": [-0.02, -0.04],
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


def test_download_multiplanet_systems_normalizes_and_saves_catalogue(
    monkeypatch, tmp_path
):
    """Test that the downloader writes raw archive fields and caches them."""
    raw_catalogue = pd.DataFrame(
        {
            "hostname": ["Star A"],
            "sy_pnum": [3],
            "pl_name": ["Planet b"],
            "discoverymethod": ["Radial Velocity"],
            "pl_controv_flag": [0],
            "st_mass": [1.2],
            "pl_orbper": [10.0],
            "pl_orbeccen": [0.1],
            "pl_msinie": [1.0],
            "st_masserr1": [0.05],
            "pl_orbpererr1": [0.5],
            "pl_orbeccenerr1": [0.02],
            "pl_msinieerr1": [0.1],
            "st_masserr2": [-0.07],
            "pl_orbpererr2": [-1.0],
            "pl_orbeccenerr2": [-0.04],
            "pl_msinieerr2": [-0.2],
        }
    )

    class FakeTable:
        def to_pandas(self):
            return raw_catalogue

    class FakeResult:
        def to_table(self):
            return FakeTable()

    class FakeService:
        def __init__(self, url):
            self.url = url

        def search(self, query):
            self.query = query
            return FakeResult()

    monkeypatch.setattr("cinemas.dataloading.vo.dal.TAPService", FakeService)

    save_path = tmp_path / "exoplanet_catalogue.csv"
    catalogue = download_multiplanet_systems(save_path=save_path, overwrite=True)

    assert list(catalogue["hostname"]) == ["Star A"]
    assert list(catalogue["pl_name"]) == ["Planet b"]
    assert list(catalogue["pl_controv_flag"]) == [0]
    assert list(catalogue["discoverymethod"]) == ["Radial Velocity"]
    assert list(catalogue["pl_msinie"]) == [1.0]
    assert save_path.exists()

    saved_catalogue = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(saved_catalogue, catalogue, check_dtype=False)


def test_download_multiplanet_systems_uses_cached_file(tmp_path, monkeypatch):
    """
    Test that the downloader returns an existing cached file without redownloading.
    """
    cached_catalogue = pd.DataFrame(
        {
            "hostname": ["Star A"],
            "sy_pnum": [3],
            "pl_name": ["Planet b"],
            "discoverymethod": ["Radial Velocity"],
            "pl_controv_flag": [0],
            "st_mass": [1.2],
            "pl_orbper": [10.0],
            "pl_orbeccen": [0.1],
            "pl_msinie": [1.0],
            "st_masserr1": [0.05],
            "pl_orbpererr1": [0.5],
            "pl_orbeccenerr1": [0.02],
            "pl_msinieerr1": [0.1],
            "st_masserr2": [-0.07],
            "pl_orbpererr2": [-1.0],
            "pl_orbeccenerr2": [-0.04],
            "pl_msinieerr2": [-0.2],
        }
    )

    save_path = tmp_path / "exoplanet_catalogue.csv"
    cached_catalogue.to_csv(save_path, index=False)

    def fail_search(*args, **kwargs):
        raise AssertionError("download should not be called when cache exists")

    monkeypatch.setattr("cinemas.dataloading.vo.dal.TAPService", fail_search)

    catalogue = download_multiplanet_systems(save_path=save_path, overwrite=False)

    pd.testing.assert_frame_equal(catalogue, cached_catalogue)
