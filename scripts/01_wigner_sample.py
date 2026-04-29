#!/usr/bin/env python
"""Generate Wigner-sampled initial conditions from a Gaussian opt+freq log."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gaussian_parser import parse_gaussian_log
from utils.io_utils import ensure_directory, write_json, write_velocity_text, write_xyz
from utils.wigner import classical_kinetic_energy_au, sample_wigner, select_modes, zero_point_energy_au


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-log", default="opt_freq/Trip_C18_opt.log", help="Gaussian opt+freq log file")
    parser.add_argument("-n", "--n-samples", type=int, required=True, help="Number of Wigner samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="samples", help="Output directory for sampled trajectories")
    parser.add_argument(
        "--low-freq-cutoff-cm1",
        type=float,
        default=None,
        help="Optional absolute-frequency cutoff; modes below this threshold are skipped.",
    )
    parser.add_argument(
        "--keep-imaginary",
        action="store_true",
        help="Keep imaginary modes instead of dropping them.",
    )
    parser.add_argument(
        "--bond",
        nargs=2,
        type=int,
        metavar=("I", "J"),
        default=(1, 2),
        help="One-based atom indices for the diagnostic bond-length histogram.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to reuse it.")

    ensure_directory(output_dir)
    data = parse_gaussian_log(args.input_log)
    active_freqs, _, _, active_mask = select_modes(
        data.frequencies_cm1,
        data.reduced_masses_amu,
        data.normal_modes,
        low_frequency_cutoff_cm1=args.low_freq_cutoff_cm1,
        drop_imaginary=not args.keep_imaginary,
    )
    if active_freqs.size == 0:
        raise ValueError("No vibrational modes remain after applying the mode filter.")

    rng = np.random.default_rng(args.seed)
    samples = []
    ke_values_au = []
    positions = []
    bond_lengths = []
    bond_i, bond_j = args.bond[0] - 1, args.bond[1] - 1

    for sample_idx in range(args.n_samples):
        sample = sample_wigner(
            data,
            rng=rng,
            low_frequency_cutoff_cm1=args.low_freq_cutoff_cm1,
            drop_imaginary=not args.keep_imaginary,
        )
        sample_dir = ensure_directory(output_dir / f"traj_{sample_idx:04d}")
        comment = f"seed={args.seed}; sample_idx={sample_idx}"
        write_xyz(sample_dir / "init.xyz", data.symbols, sample.positions_ang, comment=comment)
        np.save(sample_dir / "init.vel.npy", sample.velocities_ang_fs.astype(np.float64))
        write_velocity_text(sample_dir / "init.vel", sample.velocities_ang_fs)

        samples.append(sample)
        positions.append(sample.positions_ang)
        ke_values_au.append(classical_kinetic_energy_au(data.masses_amu, sample.velocities_ang_fs))
        bond_lengths.append(float(np.linalg.norm(sample.positions_ang[bond_i] - sample.positions_ang[bond_j])))

    mean_structure = np.mean(np.stack(positions, axis=0), axis=0)
    zpe_au = zero_point_energy_au(active_freqs)
    validation = {
        "parser": data.parser,
        "n_atoms": data.n_atoms,
        "n_modes_total": int(data.n_modes),
        "n_modes_active": int(active_freqs.size),
        "seed": args.seed,
        "mean_abs_coordinate_error_ang": float(np.mean(np.abs(mean_structure - data.equilibrium_xyz_ang))),
        "max_abs_coordinate_error_ang": float(np.max(np.abs(mean_structure - data.equilibrium_xyz_ang))),
        "mean_kinetic_energy_au": float(np.mean(ke_values_au)),
        "expected_mean_kinetic_energy_au": float(0.5 * zpe_au),
        "zero_point_energy_au": float(zpe_au),
    }

    manifest = {
        "source_log": str(Path(args.input_log).as_posix()),
        "method": "wigner",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "parser": data.parser,
        "atom_order": data.symbols,
        "units": {"position": "angstrom", "velocity": "angstrom/fs"},
        "mode_filter": {
            "drop_imaginary": not args.keep_imaginary,
            "low_freq_cutoff_cm1": args.low_freq_cutoff_cm1,
            "n_modes_active": int(active_freqs.size),
        },
        "validation": validation,
    }
    write_json(output_dir / "manifest.json", manifest)

    histogram_path = output_dir / "bond_histogram.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(bond_lengths, bins=min(40, max(10, args.n_samples // 2)), color="#1f77b4", alpha=0.85)
    ax.set_xlabel(f"Bond length ({args.bond[0]}-{args.bond[1]}) / angstrom")
    ax.set_ylabel("Count")
    ax.set_title("Wigner diagnostic bond-length histogram")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=200)
    plt.close(fig)

    write_json(output_dir / "validation.json", validation)
    print(f"Wrote {args.n_samples} samples to {output_dir}")
    print(f"Validation summary: {validation}")


if __name__ == "__main__":
    main()
