"""
cinemas.py
==========
Constraining INclinations of Exoplanets and their MAsses by Stability
---------------------------------------------------------------------
A framework for obtaining Bayesian constraints on the inclinations, and hence true
masses (as well as other orbital parameters) of exoplanets in multi-planet systems by
using the constraint of long-term dynamical stability. The framework uses the stability
classifier from SPOCK. The result? Absolute cinema.
"""

from pathlib import Path

import numpy as np

from . import dataloading, mcmc


def main():

    COMPACT_MULTIPLANET_RV_SYSTEMS = [
        # "Barnard's star",
        # "GJ 667 C",  # Can't find valid walkers??
        # "HD 158259",  # Taking ages??
        "HD 184010",
        "HD 215152",
        "HD 28471",
        "HD 34445",
        "HD 38677",
        "HD 40307",
        "YZ Cet",
    ]

    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent

    DATA_DIR = ROOT_DIR / "data"
    EXOPLANET_CATALOGUE_PATH = DATA_DIR / "exoplanet.eu_catalog_15-03-26_22_54_01.csv"

    RESULTS_DIR = ROOT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for system in COMPACT_MULTIPLANET_RV_SYSTEMS:
        print("\n" + "=" * 50)
        print(f"In CINEMAS now: {system}...")

        results_path = RESULTS_DIR / f"{system.lower().replace(' ', '_')}_10k.npz"

        np.random.seed(42)

        system_obs = dataloading.load_system_observations(
            system, EXOPLANET_CATALOGUE_PATH
        )

        try:
            samples, tau, acceptance_fraction = mcmc.run_mcmc_sampling(
                system_obs,
                nsteps=10000,
                nwalkers=None,  # Use default number of walkers
            )
        except ValueError as e:
            print(f"Error during MCMC sampling for {system}: {e}")
            continue

        np.savez_compressed(
            results_path,
            samples=samples,
            tau=tau,
            acceptance_fraction=acceptance_fraction,
        )
        print(f"Results saved to {results_path.resolve()} .")
        print(f"Autocorrelation time: {tau}")
        print(f"Acceptance fraction: {acceptance_fraction}")

        print("\n" + "=" * 50 + "\n\n")


if __name__ == "__main__":
    main()
