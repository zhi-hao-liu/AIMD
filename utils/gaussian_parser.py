"""Gaussian opt+freq parser with an optional cclib fast path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from utils.constants import ATOMIC_SYMBOLS


_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[DEde][-+]?\d+)?|[-+]?\d+(?:[DEde][-+]?\d+)")


@dataclass
class GaussianFrequencyData:
    symbols: list[str]
    atomic_numbers: np.ndarray
    masses_amu: np.ndarray
    equilibrium_xyz_ang: np.ndarray
    frequencies_cm1: np.ndarray
    reduced_masses_amu: np.ndarray
    normal_modes: np.ndarray
    parser: str

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def n_modes(self) -> int:
        return int(self.frequencies_cm1.shape[0])


def parse_gaussian_log(path: str | Path, prefer_cclib: bool = True) -> GaussianFrequencyData:
    path = Path(path)
    if prefer_cclib:
        try:
            return _parse_with_cclib(path)
        except Exception:
            pass
    return _parse_with_text(path)


def _parse_with_cclib(path: Path) -> GaussianFrequencyData:
    from cclib import io as cclib_io

    data = cclib_io.ccread(str(path))
    if data is None:
        raise ValueError(f"cclib could not parse {path}")

    atomic_numbers = np.asarray(data.atomnos, dtype=int)
    symbols = [ATOMIC_SYMBOLS[int(number)] for number in atomic_numbers]

    masses_attr = getattr(data, "atommasses", None)
    if masses_attr is None:
        masses_amu = np.array([12.0 if symbol == "C" else 1.007825 for symbol in symbols], dtype=float)
    else:
        masses_amu = np.asarray(masses_attr, dtype=float)

    coords = np.asarray(data.atomcoords[-1], dtype=float)
    frequencies = np.asarray(data.vibfreqs, dtype=float)
    reduced_masses = np.asarray(getattr(data, "vibrmasses"), dtype=float)
    vibdisps = np.asarray(data.vibdisps, dtype=float)
    normal_modes = np.transpose(vibdisps, (1, 2, 0))

    return GaussianFrequencyData(
        symbols=symbols,
        atomic_numbers=atomic_numbers,
        masses_amu=masses_amu,
        equilibrium_xyz_ang=coords,
        frequencies_cm1=frequencies,
        reduced_masses_amu=reduced_masses,
        normal_modes=normal_modes,
        parser="cclib",
    )


def _parse_with_text(path: Path) -> GaussianFrequencyData:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atomic_numbers, coords = _parse_last_standard_orientation(lines)
    masses_amu = _parse_atomic_masses(lines, len(atomic_numbers))
    frequencies, reduced_masses, normal_modes = _parse_frequency_blocks(lines, len(atomic_numbers))
    symbols = [ATOMIC_SYMBOLS[int(number)] for number in atomic_numbers]

    return GaussianFrequencyData(
        symbols=symbols,
        atomic_numbers=atomic_numbers,
        masses_amu=masses_amu,
        equilibrium_xyz_ang=coords,
        frequencies_cm1=frequencies,
        reduced_masses_amu=reduced_masses,
        normal_modes=normal_modes,
        parser="text",
    )


def _parse_last_standard_orientation(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    indices = [idx for idx, line in enumerate(lines) if "Standard orientation:" in line]
    if not indices:
        raise ValueError("Could not find a Gaussian 'Standard orientation' block.")

    start = indices[-1] + 5
    atomic_numbers: list[int] = []
    coords: list[list[float]] = []

    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("-----"):
            break
        parts = stripped.split()
        if len(parts) < 6:
            continue
        atomic_numbers.append(int(parts[1]))
        coords.append([float(parts[3]), float(parts[4]), float(parts[5])])

    if not atomic_numbers:
        raise ValueError("Failed to parse coordinates from the final Standard orientation block.")

    return np.asarray(atomic_numbers, dtype=int), np.asarray(coords, dtype=float)


def _parse_atomic_masses(lines: list[str], n_atoms: int) -> np.ndarray:
    masses: list[float] = []
    for line in lines:
        if "AtmWgt=" in line:
            masses.extend(_extract_floats(line))
    if len(masses) < n_atoms:
        raise ValueError(f"Expected at least {n_atoms} atomic masses, found {len(masses)}")
    return np.asarray(masses[-n_atoms:], dtype=float)


def _parse_frequency_blocks(lines: list[str], n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies: list[float] = []
    reduced_masses: list[float] = []
    mode_columns: list[np.ndarray] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "Frequencies --" not in line:
            idx += 1
            continue

        freq_values = _extract_floats(line)
        red_mass_values = _extract_floats(lines[idx + 1])
        if len(freq_values) != len(red_mass_values):
            raise ValueError("Frequency and reduced-mass columns are misaligned in Gaussian output.")

        header_idx = idx + 4
        atom_rows = lines[header_idx + 1 : header_idx + 1 + n_atoms]
        if len(atom_rows) != n_atoms:
            raise ValueError("Incomplete normal-coordinate block in Gaussian output.")

        n_block_modes = len(freq_values)
        block = np.zeros((n_atoms, 3, n_block_modes), dtype=float)
        for atom_idx, atom_line in enumerate(atom_rows):
            parts = atom_line.split()
            values = [float(token.replace("D", "E").replace("d", "e")) for token in parts[2:]]
            if len(values) != 3 * n_block_modes:
                raise ValueError("Unexpected Gaussian normal-coordinate row width.")
            for mode_idx in range(n_block_modes):
                block[atom_idx, :, mode_idx] = values[3 * mode_idx : 3 * (mode_idx + 1)]

        for mode_idx, frequency in enumerate(freq_values):
            frequencies.append(frequency)
            reduced_masses.append(red_mass_values[mode_idx])
            mode_columns.append(block[:, :, mode_idx])

        idx = header_idx + 1 + n_atoms

    if not frequencies:
        raise ValueError("No vibrational frequencies were found in the Gaussian log.")

    modes = np.stack(mode_columns, axis=-1)
    return (
        np.asarray(frequencies, dtype=float),
        np.asarray(reduced_masses, dtype=float),
        modes,
    )


def _extract_floats(text: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(text)]

