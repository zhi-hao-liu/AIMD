"""Filesystem helpers for XYZ, JSON, numpy arrays, and xTB restart files."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterator

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_xyz(path: str | Path, symbols: list[str], coords_ang: np.ndarray, comment: str = "") -> None:
    path = Path(path)
    lines = [str(len(symbols)), comment]
    for symbol, coord in zip(symbols, coords_ang):
        lines.append(f"{symbol:2s} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray, str]:
    frames = list(iter_xyz_frames(path))
    if len(frames) != 1:
        raise ValueError(f"Expected a single XYZ frame in {path}, found {len(frames)}")
    return frames[0]


def iter_xyz_frames(path: str | Path) -> Iterator[tuple[list[str], np.ndarray, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            n_atoms = int(stripped)
            comment = handle.readline().rstrip("\n")
            symbols: list[str] = []
            coords = np.zeros((n_atoms, 3), dtype=float)
            for atom_idx in range(n_atoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line in {path}")
                symbols.append(parts[0])
                coords[atom_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
            yield symbols, coords, comment


def read_last_xyz_frame(path: str | Path) -> tuple[list[str], np.ndarray, str]:
    last_frame = None
    for frame in iter_xyz_frames(path):
        last_frame = frame
    if last_frame is None:
        raise ValueError(f"No XYZ frames found in {path}")
    return last_frame


def write_velocity_text(path: str | Path, velocities_ang_fs: np.ndarray) -> None:
    path = Path(path)
    lines = ["# angstrom/fs"]
    for velocity in velocities_ang_fs:
        lines.append(f"{velocity[0]: .12e} {velocity[1]: .12e} {velocity[2]: .12e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_charges(path: str | Path) -> np.ndarray:
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                values.append(float(stripped))
    return np.asarray(values, dtype=float)


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_mdrestart(path: str | Path, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    numeric_rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                values = [float(token.replace("D", "E").replace("d", "e")) for token in parts[:6]]
            except ValueError:
                continue
            numeric_rows.append(values)
    if len(numeric_rows) < n_atoms:
        raise ValueError(f"Could not recover {n_atoms} restart rows from {path}")
    array = np.asarray(numeric_rows[-n_atoms:], dtype=float)
    return array[:, :3], array[:, 3:]

