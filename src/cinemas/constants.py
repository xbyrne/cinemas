"""
constants.py
============
Constants for the CINEMAS script.
"""

from pathlib import Path

# ---------------------------
# Constants for data sourcing

DISCRETE_FIELDS = [
    "hostname",
    "sy_pnum",
    "pl_name",
    "discoverymethod",
    "pl_controv_flag",
]
CONTINUOUS_FIELDS = ["st_mass", "pl_orbper", "pl_orbeccen", "pl_msinie"]
ALL_FIELDS = (
    DISCRETE_FIELDS
    + CONTINUOUS_FIELDS
    + [f"{field}err1" for field in CONTINUOUS_FIELDS]
    + [f"{field}err2" for field in CONTINUOUS_FIELDS]
)

DEFAULT_DOWNLOAD_PATH = (
    Path(__file__).parent.parent.parent / "data" / "exoplanet_catalogue.csv"
).resolve()

# ------------------
# Physical constants

MSUN_MEARTH = 333_000

# ------
# Priors

I_MIN, I_MAX = 0.1, 90.0
