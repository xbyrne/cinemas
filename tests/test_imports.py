"""
test_imports.py
===============
Smoke tests to ensure that all imports work correctly.
"""


def test_package_version():
    """Test that the package version can be imported."""
    import cinemas

    assert cinemas.__version__


def test_import_constants():
    """Test that the constants can be imported."""
    from cinemas import constants

    assert constants.ALL_FIELDS
    assert constants.DEFAULT_DOWNLOAD_PATH

    assert constants.MSUN_MEARTH

    assert constants.I_MIN
    assert constants.I_MAX


def test_import_observation_classes():
    """Test that the observation classes can be imported."""
    from cinemas import observation_classes as obs

    assert obs.Observation
    assert obs.PlanetObservations
    assert obs.SystemObservations


def test_import_main_functions():
    """Test that the main functions can be imported."""
    from cinemas import (
        download_multiplanet_systems,
        load_system_observations,
        run_mcmc_sampling,
    )

    assert callable(download_multiplanet_systems)
    assert callable(load_system_observations)
    assert callable(run_mcmc_sampling)
