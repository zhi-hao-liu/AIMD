#!/usr/bin/env python
"""Analyze xTB trajectories into fragment channels and kinetic observables."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fragments import (
    assign_integer_charges,
    canonical_channel_label,
    centered_com_positions,
    determine_terminal_state,
    fragment_formula,
    kinetic_energies_by_fragment,
)
from utils.constants import BOHR_PER_AUT_TO_ANGSTROM_PER_FS
from utils.io_utils import ensure_directory, iter_xyz_frames, read_charges, read_json, read_mdrestart, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Directory containing per-trajectory xTB runs")
    parser.add_argument("--analysis-dir", default="analysis", help="Output directory for analysis artifacts")
    parser.add_argument("--expected-total-charge", type=int, default=2, help="Expected total molecular charge")
    parser.add_argument("--stability-window-fs", type=float, default=1000.0, help="Terminal-state stability window")
    parser.add_argument("--separation-threshold-ang", type=float, default=5.0, help="Distance threshold for clean separation")
    parser.add_argument("--bond-scale", type=float, default=1.3, help="Connectivity threshold scale factor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    analysis_dir = ensure_directory(args.analysis_dir)
    per_traj_dir = ensure_directory(analysis_dir / "per_traj")
    plots_dir = ensure_directory(analysis_dir / "plots")

    fragment_rows: list[dict] = []
    channel_counter: Counter[str] = Counter()
    channel_examples: dict[str, list[dict]] = {}
    breakup_times: list[float] = []
    charged_fragment_kers: list[float] = []

    for run_dir in sorted(path for path in runs_dir.glob("traj_*") if path.is_dir()):
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        status = read_json(status_path)
        if status.get("status") != "success":
            continue

        trajectory_path = run_dir / "xtb.trj"
        charges_path = run_dir / "final_sp" / "charges"
        mdrestart_path = run_dir / "mdrestart"
        md_config_path = run_dir / "md.yaml"
        if not all(path.exists() for path in [trajectory_path, charges_path, mdrestart_path, md_config_path]):
            continue

        md_config = yaml.safe_load(md_config_path.read_text(encoding="utf-8"))
        dump_every_fs = float(md_config["md"]["dump_every_fs"])
        frames = list(iter_xyz_frames(trajectory_path))
        symbols = frames[0][0]
        frame_coords = [coords for _, coords, _ in frames]
        masses_amu = np.array(_infer_masses(symbols), dtype=float)
        charges = read_charges(charges_path)
        if charges.shape[0] != len(symbols):
            raise ValueError(f"Charge count mismatch in {charges_path}")
        _, final_velocities_au = read_mdrestart(mdrestart_path, len(symbols))
        final_velocities_ang_fs = final_velocities_au * BOHR_PER_AUT_TO_ANGSTROM_PER_FS

        terminal_state = determine_terminal_state(
            symbols=symbols,
            trajectory_frames_ang=frame_coords,
            dump_every_fs=dump_every_fs,
            stability_window_fs=args.stability_window_fs,
            separation_threshold_ang=args.separation_threshold_ang,
            bond_scale=args.bond_scale,
        )

        final_components = terminal_state.final_components
        float_fragment_charges = np.array([np.sum(charges[component]) for component in final_components], dtype=float)
        if abs(float(np.sum(float_fragment_charges)) - args.expected_total_charge) > 1.0e-3:
            raise ValueError(f"Charge conservation failed for {run_dir.name}")
        integer_charges = assign_integer_charges(float_fragment_charges, args.expected_total_charge)
        ke_trans, ke_internal, _ = kinetic_energies_by_fragment(masses_amu, final_velocities_ang_fs, final_components)
        com_positions = centered_com_positions(masses_amu, frame_coords[-1], final_components)

        fragments = []
        for frag_idx, component in enumerate(final_components):
            formula = fragment_formula(symbols, component)
            mass_amu = float(np.sum(masses_amu[np.asarray(component, dtype=int)]))
            fragment = {
                "atom_indices": [index + 1 for index in component],
                "formula": formula,
                "charge": int(integer_charges[frag_idx]),
                "float_charge": float(float_fragment_charges[frag_idx]),
                "mass_amu": mass_amu,
                "ke_trans_eV": float(ke_trans[frag_idx]),
                "ke_internal_kin_eV": float(ke_internal[frag_idx]),
                "com_position_ang": [float(value) for value in com_positions[frag_idx]],
                "com_distance_ang": float(np.linalg.norm(com_positions[frag_idx])),
            }
            fragments.append(fragment)
            fragment_rows.append(
                {
                    "traj_id": run_dir.name,
                    "formula": formula,
                    "charge": fragment["charge"],
                    "mass_amu": f"{mass_amu:.6f}",
                    "ke_trans_eV": f"{fragment['ke_trans_eV']:.6f}",
                    "ke_internal_kin_eV": f"{fragment['ke_internal_kin_eV']:.6f}",
                    "t_breakup_fs": f"{terminal_state.t_breakup_fs:.2f}",
                    "status": terminal_state.status,
                }
            )
            if fragment["charge"] != 0:
                charged_fragment_kers.append(fragment["ke_trans_eV"])

        channel = canonical_channel_label(fragments)
        breakup_times.append(terminal_state.t_breakup_fs)
        if terminal_state.status == "converged":
            channel_counter[channel] += 1
            channel_examples[channel] = fragments

        per_traj_payload = {
            "traj_id": run_dir.name,
            "terminal_state": {
                "status": terminal_state.status,
                "t_breakup_fs": terminal_state.t_breakup_fs,
                "min_interfragment_distance_ang": terminal_state.min_interfragment_distance_ang,
                "connectivity_stable": terminal_state.connectivity_stable,
                "geometrically_separated": terminal_state.geometrically_separated,
            },
            "channel": channel,
            "fragments": fragments,
        }
        write_json(per_traj_dir / f"{run_dir.name}.json", per_traj_payload)

    write_fragments_summary(analysis_dir / "fragments_summary.csv", fragment_rows)
    write_channels_csv(analysis_dir / "channels.csv", channel_counter, channel_examples)
    write_plots(plots_dir, channel_counter, breakup_times, charged_fragment_kers)
    print(f"Wrote analysis to {analysis_dir}")


def _infer_masses(symbols: list[str]) -> list[float]:
    masses = {"H": 1.007825, "C": 12.0, "N": 14.003074, "O": 15.994915, "S": 31.972071}
    return [masses.get(symbol, 12.0) for symbol in symbols]


def write_fragments_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "traj_id",
            "formula",
            "charge",
            "mass_amu",
            "ke_trans_eV",
            "ke_internal_kin_eV",
            "t_breakup_fs",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_channels_csv(path: Path, channel_counter: Counter[str], channel_examples: dict[str, list[dict]]) -> None:
    total = sum(channel_counter.values())
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["channel", "count", "probability", "example_fragments"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for channel, count in channel_counter.most_common():
            probability = (count / total) if total else 0.0
            writer.writerow(
                {
                    "channel": channel,
                    "count": count,
                    "probability": f"{probability:.6f}",
                    "example_fragments": "; ".join(
                        f"{fragment['formula']}[{fragment['charge']:+d}]"
                        for fragment in channel_examples[channel]
                    ),
                }
            )


def write_plots(plots_dir: Path, channel_counter: Counter[str], breakup_times: list[float], charged_fragment_kers: list[float]) -> None:
    if channel_counter:
        labels, counts = zip(*channel_counter.most_common(10))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(labels)), counts, color="#d97706")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Top Fragmentation Channels")
        fig.tight_layout()
        fig.savefig(plots_dir / "channel_branching.png", dpi=200)
        plt.close(fig)

    if charged_fragment_kers:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(charged_fragment_kers, bins=min(40, max(10, len(charged_fragment_kers) // 2)), color="#2563eb", alpha=0.85)
        ax.set_xlabel("Charged-fragment translational KE / eV")
        ax.set_ylabel("Count")
        ax.set_title("Charged Fragment KER Distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "charged_fragment_ker.png", dpi=200)
        plt.close(fig)

    if breakup_times:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(breakup_times, bins=min(40, max(10, len(breakup_times) // 2)), color="#059669", alpha=0.85)
        ax.set_xlabel("Breakup time / fs")
        ax.set_ylabel("Count")
        ax.set_title("Breakup Time Distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "breakup_times.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
