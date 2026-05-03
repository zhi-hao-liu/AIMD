from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.gaussian_parser import (
    GaussianFrequencyData,
    _coerce_normal_modes,
    parse_gaussian_log,
)
from utils.wigner import sample_wigner


def test_parse_trip_c18_text_shapes() -> None:
    data = parse_gaussian_log(Path("opt_freq/Trip_C18_opt.log"), prefer_cclib=False)

    assert data.n_atoms == 30
    assert data.masses_amu.shape == (30,)
    assert data.equilibrium_xyz_ang.shape == (30, 3)
    assert data.frequencies_cm1.shape == (84,)
    assert data.reduced_masses_amu.shape == (84,)
    assert data.normal_modes.shape == (30, 3, 84)


def test_coerce_normal_modes_handles_cclib_axis_order() -> None:
    n_atoms = 2
    n_modes = 4
    target = np.arange(n_atoms * 3 * n_modes, dtype=float).reshape(n_atoms, 3, n_modes)
    cclib_style = np.transpose(target, (2, 0, 1))

    coerced = _coerce_normal_modes(cclib_style, n_atoms=n_atoms, n_modes=n_modes)

    assert coerced.shape == target.shape
    assert np.array_equal(coerced, target)


def test_sample_wigner_accepts_non_flat_mass_input() -> None:
    data = GaussianFrequencyData(
        symbols=["H", "H"],
        atomic_numbers=np.array([1, 1], dtype=int),
        masses_amu=np.array([[1.007825], [1.007825]], dtype=float),
        equilibrium_xyz_ang=np.zeros((2, 3), dtype=float),
        frequencies_cm1=np.array([1000.0], dtype=float),
        reduced_masses_amu=np.array([1.0], dtype=float),
        normal_modes=np.array(
            [
                [[1.0], [0.0], [0.0]],
                [[-1.0], [0.0], [0.0]],
            ],
            dtype=float,
        ),
        parser="test",
    )

    sample = sample_wigner(data, rng=np.random.default_rng(0))

    assert sample.positions_ang.shape == (2, 3)
    assert sample.velocities_ang_fs.shape == (2, 3)
