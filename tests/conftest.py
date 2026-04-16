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
        minimum_mass=Observation(1.0, 0.1),
        period=Observation(10.0, 0.5),
        eccentricity=Observation(0.1, 0.02),
    )
    planet_c = PlanetObservations(
        name="c",
        minimum_mass=Observation(2.0, 0.2),
        period=Observation(20.0, 1.0),
        eccentricity=Observation(0.2, 0.04),
    )

    return SystemObservations(
        star_name="Star A",
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
            "name": "Raxacoricofallapatorius",
            "mass_sini": 1.0,
            "mass_sini_error_min": 0.1,
            "mass_sini_error_max": 0.2,
            "orbital_period": 10.0,
            "orbital_period_error_min": 0.5,
            "orbital_period_error_max": 1.0,
            "eccentricity": 0.1,
            "eccentricity_error_min": 0.02,
            "eccentricity_error_max": 0.04,
        }
    )


@pytest.fixture
def example_planet_data_row_without_eccentricity() -> pd.Series:
    """A single row of planet data without eccentricity, as a pandas Series."""
    return pd.Series(
        {
            "name": "Raxacoricovarlonpatorius",
            "mass_sini": 1.0,
            "mass_sini_error_min": 0.1,
            "mass_sini_error_max": 0.2,
            "orbital_period": 10.0,
            "orbital_period_error_min": 0.5,
            "orbital_period_error_max": 1.0,
            "eccentricity": None,
            "eccentricity_error_min": None,
            "eccentricity_error_max": None,
        }
    )


@pytest.fixture
def example_system_data_compact() -> pd.DataFrame:
    """A minimal DataFrame representing a compact multiplanet system."""
    return pd.DataFrame(
        {
            "name": ["Planet b", "Planet c", "Planet d"],
            "star_mass": [1.0, 1.0, 1.0],
            "star_mass_error_min": [0.05, 0.05, 0.05],
            "star_mass_error_max": [0.05, 0.05, 0.05],
            "mass_sini": [1.0, 2.0, 3.0],
            "mass_sini_error_min": [0.1, 0.2, 0.3],
            "mass_sini_error_max": [0.1, 0.2, 0.3],
            "orbital_period": [10.0, 25.0, 15.0],
            # ^ NB: Intentionally out of order to test sorting
            "orbital_period_error_min": [0.5, 1.0, 0.5],
            "orbital_period_error_max": [0.5, 1.0, 0.5],
            "eccentricity": [0.1, 0.2, 0.3],
            "eccentricity_error_min": [0.02, 0.04, 0.06],
            "eccentricity_error_max": [0.02, 0.04, 0.06],
            "detection_type": ["Radial Velocity", "Radial Velocity", "Radial Velocity"],
            "star_name": ["Star A", "Star A", "Star A"],
        }
    )


@pytest.fixture
def example_system_data_non_compact(example_system_data_compact) -> pd.DataFrame:
    """A minimal DataFrame representing a non-compact multiplanet system."""
    # Start with the compact system data
    non_compact_data = example_system_data_compact.copy()

    # Modify the orbital periods to make it non-compact
    non_compact_data["orbital_period"] = [50.0, 15.0, 10.0]
    # (again, intentionally out of order to test sorting)

    # Modify also the star name
    non_compact_data["star_name"] = ["Star B", "Star B", "Star B"]

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
