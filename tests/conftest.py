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
