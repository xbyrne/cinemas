"""
conftest.py
===========
Fixtures for testing the CINEMAS package.
"""

import pandas as pd
import pytest

from cinemas.observation_classes import (
    Observation,
    PlanetObservations,
    SystemObservations,
)

# ============
# Observations


@pytest.fixture
def simple_observation() -> Observation:
    """A minimal observation"""
    return Observation(mean=1.0, error=0.1)


@pytest.fixture
def simple_planet_observations() -> PlanetObservations:
    """A minimal set of planet observations."""
    return PlanetObservations(
        name="Planet b",
        minimum_mass=Observation(mean=1.0, error=0.1),
        period=Observation(mean=10.0, error=0.5),
        eccentricity=Observation(mean=0.1, error=0.02),
    )


@pytest.fixture
def simple_system_observations() -> SystemObservations:
    """A minimal set of system observations."""
    star_mass = Observation(mean=1.0, error=0.05)

    planet_b = PlanetObservations(
        name="b",
        minimum_mass=Observation(mean=1.0, error=0.1),
        period=Observation(mean=10.0, error=0.5),
        eccentricity=Observation(mean=0.1, error=0.02),
    )
    planet_c = PlanetObservations(
        name="c",
        minimum_mass=Observation(mean=2.0, error=0.2),
        period=Observation(mean=20.0, error=1.0),
        eccentricity=Observation(mean=0.2, error=0.04),
    )

    return SystemObservations(
        star_name="Star A",
        star_mass=star_mass,
        planet_observations=[planet_b, planet_c],
    )


@pytest.fixture
def mixed_system_observations() -> SystemObservations:
    """A two-planet system with mixed Gaussian and Uniform prior families."""
    star_mass = Observation(mean=0.9, error=0.03)

    planet_b = PlanetObservations(
        name="b",
        minimum_mass=Observation(distribution="uniform", bounds=(0.5, 1.5)),
        period=Observation(distribution="uniform", bounds=(8.0, 12.0)),
        eccentricity=Observation(distribution="uniform", bounds=(0.0, 0.3)),
    )
    planet_c = PlanetObservations(
        name="c",
        minimum_mass=Observation(mean=2.0, error=0.2),
        period=Observation(mean=20.0, error=1.0),
        eccentricity=Observation(mean=0.2, error=0.04),
    )

    return SystemObservations(
        star_name="Mixed Star",
        star_mass=star_mass,
        planet_observations=[planet_b, planet_c],
    )


@pytest.fixture
def mixed_uniform_star_mass_system_observations() -> SystemObservations:
    """A mixed-prior system where star mass prior is also Uniform."""
    star_mass = Observation(distribution="uniform", bounds=(0.8, 1.2))

    planet_b = PlanetObservations(
        name="b",
        minimum_mass=Observation(distribution="uniform", bounds=(0.3, 0.8)),
        period=Observation(mean=10.0, error=0.5),
        eccentricity=Observation(distribution="uniform", bounds=(0.0, 0.25)),
    )
    planet_c = PlanetObservations(
        name="c",
        minimum_mass=Observation(mean=1.6, error=0.15),
        period=Observation(distribution="uniform", bounds=(18.0, 22.0)),
        eccentricity=Observation(mean=0.15, error=0.03),
    )

    return SystemObservations(
        star_name="Mixed Star (Uniform M*)",
        star_mass=star_mass,
        planet_observations=[planet_b, planet_c],
    )


# ============
# Data loading


@pytest.fixture
def example_planet_data_row() -> pd.Series:
    """A single row of planet data, as a pandas Series."""
    return pd.Series(
        {
            "pl_name": "Raxacoricofallapatorius",
            "pl_msinie": 1.0,
            "pl_msinieerr1": 0.1,
            "pl_msinieerr2": -0.2,
            "pl_orbper": 10.0,
            "pl_orbpererr1": 0.5,
            "pl_orbpererr2": -1.0,
            "pl_orbeccen": 0.1,
            "pl_orbeccenerr1": 0.02,
            "pl_orbeccenerr2": -0.04,
        }
    )


@pytest.fixture
def example_planet_data_row_without_eccentricity() -> pd.Series:
    """A single row of planet data without eccentricity, as a pandas Series."""
    return pd.Series(
        {
            "pl_name": "Raxacoricovarlonpatorius",
            "pl_msinie": 1.0,
            "pl_msinieerr1": 0.1,
            "pl_msinieerr2": -0.2,
            "pl_orbper": 10.0,
            "pl_orbpererr1": 0.5,
            "pl_orbpererr2": -1.0,
            "pl_orbeccen": None,
            "pl_orbeccenerr1": None,
            "pl_orbeccenerr2": None,
        }
    )


@pytest.fixture
def example_system_data_compact() -> pd.DataFrame:
    """A minimal DataFrame representing a compact multiplanet system."""
    return pd.DataFrame(
        {
            "pl_name": ["Planet b", "Planet c", "Planet d"],
            "st_mass": [1.0, 1.0, 1.0],
            "st_masserr1": [0.05, 0.05, 0.05],
            "st_masserr2": [-0.05, -0.05, -0.05],
            "pl_msinie": [1.0, 2.0, 3.0],
            "pl_msinieerr1": [0.1, 0.2, 0.3],
            "pl_msinieerr2": [-0.1, -0.2, -0.3],
            "pl_orbper": [10.0, 25.0, 15.0],
            # ^ NB: Intentionally out of order to test sorting
            "pl_orbpererr1": [0.5, 1.0, 0.5],
            "pl_orbpererr2": [-0.5, -1.0, -0.5],
            "pl_orbeccen": [0.1, 0.2, 0.3],
            "pl_orbeccenerr1": [0.02, 0.04, 0.06],
            "pl_orbeccenerr2": [-0.02, -0.04, -0.06],
            "pl_controv_flag": [0, 0, 0],
            "discoverymethod": [
                "Radial Velocity",
                "Radial Velocity",
                "Radial Velocity",
            ],
            "hostname": ["Star A", "Star A", "Star A"],
        }
    )


@pytest.fixture
def example_system_data_non_compact(example_system_data_compact) -> pd.DataFrame:
    """A minimal DataFrame representing a non-compact multiplanet system."""
    # Start with the compact system data
    non_compact_data = example_system_data_compact.copy()

    # Modify the orbital periods to make it non-compact
    non_compact_data["pl_orbper"] = [50.0, 15.0, 10.0]
    # (again, intentionally out of order to test sorting)

    # Modify also the star name
    non_compact_data["hostname"] = ["Star B", "Star B", "Star B"]

    return non_compact_data


@pytest.fixture
def example_exoplanet_catalogue(
    example_system_data_compact: pd.DataFrame,
    example_system_data_non_compact: pd.DataFrame,
) -> pd.DataFrame:
    """
    A minimal DataFrame representing an exoplanet catalogue with two systems.
    One system is compact, and the other is non-compact.
    """
    return pd.concat(
        [example_system_data_compact, example_system_data_non_compact],
        ignore_index=True,
    )
