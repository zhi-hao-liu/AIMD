"""Physical constants and lightweight element tables used by the AIMD workflow."""

from __future__ import annotations

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

AU_TIME_TO_FS = 0.02418884326505
FS_TO_AU_TIME = 1.0 / AU_TIME_TO_FS

AMU_TO_AU_MASS = 1822.888486209
AU_MASS_TO_AMU = 1.0 / AMU_TO_AU_MASS

HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# In atomic units, hbar = 1, so angular frequency and energy share the same numeric
# conversion from wavenumber.
WAVENUMBER_TO_AU_FREQ = 4.556335252912e-6
AU_FREQ_TO_WAVENUMBER = 1.0 / WAVENUMBER_TO_AU_FREQ

ANGSTROM_PER_FS_TO_BOHR_PER_AUT = ANGSTROM_TO_BOHR * AU_TIME_TO_FS
BOHR_PER_AUT_TO_ANGSTROM_PER_FS = 1.0 / ANGSTROM_PER_FS_TO_BOHR_PER_AUT

ATOMIC_SYMBOLS = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    35: "Br",
    53: "I",
}

COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

