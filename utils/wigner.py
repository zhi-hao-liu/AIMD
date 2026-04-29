"""Wigner sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.constants import (
    AMU_TO_AU_MASS,
    ANGSTROM_PER_FS_TO_BOHR_PER_AUT,
    ANGSTROM_TO_BOHR,
    AU_MASS_TO_AMU,
    BOHR_PER_AUT_TO_ANGSTROM_PER_FS,
    BOHR_TO_ANGSTROM,
    WAVENUMBER_TO_AU_FREQ,
)
from utils.gaussian_parser import GaussianFrequencyData


@dataclass
class WignerSample:
    positions_ang: np.ndarray
    velocities_ang_fs: np.ndarray
    normal_coordinates_au: np.ndarray
    normal_momenta_au: np.ndarray
    kinetic_energy_au: float
    potential_energy_au: float


def normalize_modes(modes: np.ndarray) -> np.ndarray:
    flat = modes.reshape(-1, modes.shape[-1])
    norms = np.linalg.norm(flat, axis=0)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return modes / safe_norms.reshape(1, 1, -1)


def select_modes(
    frequencies_cm1: np.ndarray,
    reduced_masses_amu: np.ndarray,
    modes: np.ndarray,
    low_frequency_cutoff_cm1: float | None = None,
    drop_imaginary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones_like(frequencies_cm1, dtype=bool)
    if drop_imaginary:
        mask &= frequencies_cm1 > 0.0
    if low_frequency_cutoff_cm1 is not None:
        mask &= np.abs(frequencies_cm1) >= float(low_frequency_cutoff_cm1)
    return frequencies_cm1[mask], reduced_masses_amu[mask], modes[:, :, mask], mask


def sample_wigner(
    data: GaussianFrequencyData,
    rng: np.random.Generator,
    low_frequency_cutoff_cm1: float | None = None,
    drop_imaginary: bool = True,
    remove_com_drift: bool = True,
) -> WignerSample:
    frequencies_cm1, reduced_masses_amu, modes, _ = select_modes(
        data.frequencies_cm1,
        data.reduced_masses_amu,
        data.normal_modes,
        low_frequency_cutoff_cm1=low_frequency_cutoff_cm1,
        drop_imaginary=drop_imaginary,
    )
    modes = normalize_modes(modes)

    frequencies_au = frequencies_cm1 * WAVENUMBER_TO_AU_FREQ
    if np.any(frequencies_au <= 0.0):
        raise ValueError("Wigner sampling requires strictly positive vibrational frequencies.")
    masses_au = data.masses_amu * AMU_TO_AU_MASS
    reduced_masses_au = reduced_masses_amu * AMU_TO_AU_MASS

    sigma_q = np.sqrt(1.0 / (2.0 * reduced_masses_au * frequencies_au))
    sigma_p = np.sqrt(reduced_masses_au * frequencies_au / 2.0)
    q = rng.normal(loc=0.0, scale=sigma_q)
    p = rng.normal(loc=0.0, scale=sigma_p)

    # Gaussian normal-mode displacements from the log (and cclib's vibdisps) are
    # already Cartesian mode vectors; the reduced mass carries the mass scaling.
    delta_pos_bohr = np.sum(q.reshape(1, 1, -1) * modes, axis=-1)
    vel_bohr_aut = np.sum((p / reduced_masses_au).reshape(1, 1, -1) * modes, axis=-1)

    eq_bohr = data.equilibrium_xyz_ang * ANGSTROM_TO_BOHR
    positions_bohr = eq_bohr + delta_pos_bohr

    if remove_com_drift:
        com_velocity = np.average(vel_bohr_aut, axis=0, weights=masses_au)
        vel_bohr_aut = vel_bohr_aut - com_velocity

    positions_ang = positions_bohr * BOHR_TO_ANGSTROM
    velocities_ang_fs = vel_bohr_aut * BOHR_PER_AUT_TO_ANGSTROM_PER_FS

    kinetic_au = 0.5 * float(np.sum((p**2) / reduced_masses_au))
    potential_au = 0.5 * float(np.sum(reduced_masses_au * (frequencies_au**2) * (q**2)))

    return WignerSample(
        positions_ang=positions_ang,
        velocities_ang_fs=velocities_ang_fs,
        normal_coordinates_au=q,
        normal_momenta_au=p,
        kinetic_energy_au=kinetic_au,
        potential_energy_au=potential_au,
    )


def classical_kinetic_energy_au(masses_amu: np.ndarray, velocities_ang_fs: np.ndarray) -> float:
    masses_au = masses_amu * AMU_TO_AU_MASS
    velocities_au = velocities_ang_fs * ANGSTROM_PER_FS_TO_BOHR_PER_AUT
    return 0.5 * float(np.sum(masses_au.reshape(-1, 1) * velocities_au**2))


def zero_point_energy_au(frequencies_cm1: np.ndarray) -> float:
    positive = frequencies_cm1[frequencies_cm1 > 0.0]
    return 0.5 * float(np.sum(positive * WAVENUMBER_TO_AU_FREQ))
