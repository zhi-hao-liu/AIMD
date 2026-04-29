"""Fragment connectivity, charge assignment, and kinetic-energy analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from utils.constants import (
    AMU_TO_AU_MASS,
    ANGSTROM_PER_FS_TO_BOHR_PER_AUT,
    BOHR_TO_ANGSTROM,
    COVALENT_RADII_ANGSTROM,
    HARTREE_TO_EV,
)


@dataclass
class FragmentTerminalState:
    status: str
    final_components: list[list[int]]
    t_breakup_fs: float
    min_interfragment_distance_ang: float
    connectivity_stable: bool
    geometrically_separated: bool


def build_bond_graph(symbols: list[str], coords_ang: np.ndarray, scale: float = 1.3) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    for i, j in combinations(range(len(symbols)), 2):
        radius_i = COVALENT_RADII_ANGSTROM.get(symbols[i], 0.77)
        radius_j = COVALENT_RADII_ANGSTROM.get(symbols[j], 0.77)
        cutoff = scale * (radius_i + radius_j)
        distance = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
        if distance <= cutoff:
            graph.add_edge(i, j)
    return graph


def connected_components(symbols: list[str], coords_ang: np.ndarray, scale: float = 1.3) -> list[list[int]]:
    graph = build_bond_graph(symbols, coords_ang, scale=scale)
    return [sorted(component) for component in nx.connected_components(graph)]


def connectivity_signature(components: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(component) for component in components))


def hill_formula(symbols: list[str]) -> str:
    counts = Counter(symbols)
    order: list[str] = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    for symbol in sorted(counts):
        if symbol not in {"C", "H"}:
            order.append(symbol)

    parts = []
    for symbol in order:
        count = counts[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def fragment_formula(all_symbols: list[str], indices: list[int]) -> str:
    return hill_formula([all_symbols[idx] for idx in indices])


def assign_integer_charges(float_charges: np.ndarray, total_charge: int) -> np.ndarray:
    rounded = np.rint(float_charges).astype(int)
    delta = int(total_charge - int(rounded.sum()))
    residuals = float_charges - rounded

    if delta > 0:
        order = np.argsort(-residuals)
        for idx in order[:delta]:
            rounded[idx] += 1
    elif delta < 0:
        order = np.argsort(residuals)
        for idx in order[: -delta]:
            rounded[idx] -= 1
    return rounded


def kinetic_energies_by_fragment(
    masses_amu: np.ndarray,
    velocities_ang_fs: np.ndarray,
    components: list[list[int]],
) -> tuple[list[float], list[float], np.ndarray]:
    masses_au = masses_amu * AMU_TO_AU_MASS
    velocities_au = velocities_ang_fs * ANGSTROM_PER_FS_TO_BOHR_PER_AUT

    system_com_velocity = np.average(velocities_au, axis=0, weights=masses_au)
    centered_velocities = velocities_au - system_com_velocity

    translational: list[float] = []
    internal: list[float] = []

    for component in components:
        idx = np.asarray(component, dtype=int)
        component_masses = masses_au[idx]
        component_velocities = centered_velocities[idx]
        fragment_velocity = np.average(component_velocities, axis=0, weights=component_masses)
        ke_trans = 0.5 * float(np.sum(component_masses) * np.dot(fragment_velocity, fragment_velocity))
        residual = component_velocities - fragment_velocity
        ke_internal = 0.5 * float(np.sum(component_masses.reshape(-1, 1) * residual**2))
        translational.append(ke_trans * HARTREE_TO_EV)
        internal.append(ke_internal * HARTREE_TO_EV)

    return translational, internal, centered_velocities


def centered_com_positions(masses_amu: np.ndarray, coords_ang: np.ndarray, components: list[list[int]]) -> list[np.ndarray]:
    system_com = np.average(coords_ang, axis=0, weights=masses_amu)
    centered = coords_ang - system_com
    coms: list[np.ndarray] = []
    for component in components:
        idx = np.asarray(component, dtype=int)
        coms.append(np.average(centered[idx], axis=0, weights=masses_amu[idx]))
    return coms


def minimum_fragment_distance(coords_ang: np.ndarray, component_a: list[int], component_b: list[int]) -> float:
    subset_a = coords_ang[np.asarray(component_a, dtype=int)]
    subset_b = coords_ang[np.asarray(component_b, dtype=int)]
    distances = np.linalg.norm(subset_a[:, None, :] - subset_b[None, :, :], axis=-1)
    return float(np.min(distances))


def global_min_interfragment_distance(coords_ang: np.ndarray, components: list[list[int]]) -> float:
    if len(components) < 2:
        return 0.0
    return min(
        minimum_fragment_distance(coords_ang, components[i], components[j])
        for i, j in combinations(range(len(components)), 2)
    )


def determine_terminal_state(
    symbols: list[str],
    trajectory_frames_ang: list[np.ndarray],
    dump_every_fs: float,
    stability_window_fs: float = 1000.0,
    separation_threshold_ang: float = 5.0,
    bond_scale: float = 1.3,
) -> FragmentTerminalState:
    if not trajectory_frames_ang:
        raise ValueError("Trajectory is empty.")

    components_per_frame = [connected_components(symbols, frame, scale=bond_scale) for frame in trajectory_frames_ang]
    signatures = [connectivity_signature(components) for components in components_per_frame]
    final_components = components_per_frame[-1]
    final_signature = signatures[-1]
    min_distance_trace = [
        global_min_interfragment_distance(frame, components)
        for frame, components in zip(trajectory_frames_ang, components_per_frame)
    ]

    stable_frames = max(1, int(np.ceil(stability_window_fs / dump_every_fs)))
    tail_signatures = signatures[-stable_frames:]
    tail_distances = min_distance_trace[-stable_frames:]

    connectivity_stable = all(signature == final_signature for signature in tail_signatures)
    monotonic_separation = len(tail_distances) <= 1 or np.all(np.diff(tail_distances) >= -1.0e-3)
    geometrically_separated = (tail_distances[-1] >= separation_threshold_ang) or monotonic_separation

    if len(final_components) == 1:
        status = "converged"
    elif connectivity_stable and geometrically_separated:
        status = "converged"
    elif connectivity_stable:
        status = "unconverged_close_contact"
    else:
        status = "unconverged"

    t_breakup_fs = 0.0
    for frame_idx in range(len(signatures)):
        if all(signature == final_signature for signature in signatures[frame_idx:]):
            t_breakup_fs = frame_idx * dump_every_fs
            break

    return FragmentTerminalState(
        status=status,
        final_components=final_components,
        t_breakup_fs=t_breakup_fs,
        min_interfragment_distance_ang=min_distance_trace[-1],
        connectivity_stable=connectivity_stable,
        geometrically_separated=geometrically_separated,
    )


def canonical_channel_label(fragments: list[dict]) -> str:
    labels = [f"{fragment['formula']}[{fragment['charge']:+d}]" for fragment in fragments]
    return " + ".join(sorted(labels))
