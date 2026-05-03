"""xTB-oriented configuration and file helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import yaml

from utils.constants import ANGSTROM_PER_FS_TO_BOHR_PER_AUT, ANGSTROM_TO_BOHR


DEFAULT_MD_CONFIG = {
    "charge": 2,
    "uhf": 0,
    "gfn": 2,
    "md": {
        "time_total_fs": 5000.0,
        "step_fs": 0.5,
        "dump_every_fs": 5.0,
        "nvt": False,
        "temp_K": 0.0,
        "hmass": 1,
        "shake": 0,
        "restart": True,
    },
    "samples_dir": "samples",
    "runs_dir": "runs",
    "parallel": {
        "workers": 8,
        "retry_failed": True,
        "timeout_per_traj_s": 7200,
    },
}


def deep_merge(base: dict, update: dict) -> dict:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_md_config(path: str | Path) -> dict:
    user_config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return deep_merge(DEFAULT_MD_CONFIG, user_config)


def write_md_input(path: str | Path, config: dict) -> None:
    md = config["md"]
    time_ps = float(md["time_total_fs"]) / 1000.0
    content = "\n".join(
        [
            "$md",
            f"  nvt={'true' if md['nvt'] else 'false'}",
            f"  temp={float(md.get('temp_K', 0.0)):.6f}",
            f"  time={time_ps:.6f}",
            f"  step={float(md['step_fs']):.6f}",
            f"  dump={float(md['dump_every_fs']):.6f}",
            f"  hmass={int(md.get('hmass', 1))}",
            f"  shake={int(md.get('shake', 0))}",
            f"  restart={'true' if md.get('restart', True) else 'false'}",
            "$end",
            "",
        ]
    )
    Path(path).write_text(content, encoding="utf-8")


def render_plain_mdrestart(positions_ang: np.ndarray, velocities_ang_fs: np.ndarray) -> str:
    positions_bohr = positions_ang * ANGSTROM_TO_BOHR
    velocities_au = velocities_ang_fs * ANGSTROM_PER_FS_TO_BOHR_PER_AUT
    lines = ["-1.0"]
    for coord, velocity in zip(positions_bohr, velocities_au):
        values = [coord[0], coord[1], coord[2], velocity[0], velocity[1], velocity[2]]
        # Match xTB's fixed-width mdrestart format: header line plus 6D22.14 rows.
        lines.append("".join(f"{value:22.14E}".replace("E", "D") for value in values))
    return "\n".join(lines) + "\n"


def patch_mdrestart_template(template_text: str, positions_ang: np.ndarray, velocities_ang_fs: np.ndarray) -> str:
    formatted_rows = render_plain_mdrestart(positions_ang, velocities_ang_fs).strip().splitlines()
    lines = template_text.splitlines()

    candidate_indices = [idx for idx, line in enumerate(lines) if _looks_like_numeric_row(line)]
    if len(candidate_indices) < len(formatted_rows):
        return render_plain_mdrestart(positions_ang, velocities_ang_fs)

    start = candidate_indices[-len(formatted_rows)]
    output_lines = list(lines)
    for offset, row in enumerate(formatted_rows):
        output_lines[start + offset] = row
    return "\n".join(output_lines) + "\n"


def build_mdrestart_text(
    positions_ang: np.ndarray,
    velocities_ang_fs: np.ndarray,
    template_text: str | None = None,
) -> str:
    if template_text:
        return patch_mdrestart_template(template_text, positions_ang, velocities_ang_fs)
    return render_plain_mdrestart(positions_ang, velocities_ang_fs)


def maybe_generate_mdrestart_template(
    xtb_bin: str,
    coord_xyz: Path,
    config: dict,
    charge: int,
    uhf: int,
    gfn: int,
    timeout_s: int = 120,
) -> str | None:
    if shutil.which(xtb_bin) is None:
        return None

    md = deepcopy(config["md"])
    md["time_total_fs"] = max(float(md["step_fs"]), 0.5)
    md["dump_every_fs"] = max(float(md["step_fs"]), 0.5)
    md["restart"] = False

    with tempfile.TemporaryDirectory(prefix="xtb-template-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        probe_xyz = tmpdir_path / "coord.xyz"
        probe_xyz.write_text(coord_xyz.read_text(encoding="utf-8"), encoding="utf-8")
        probe_config = {"md": md}
        write_md_input(tmpdir_path / "md_dummy.inp", probe_config)
        with (tmpdir_path / "xtb_template.log").open("w", encoding="utf-8") as handle:
            subprocess.run(
                [
                    xtb_bin,
                    str(probe_xyz.name),
                    "--chrg",
                    str(charge),
                    "--uhf",
                    str(uhf),
                    "--gfn",
                    str(gfn),
                    "--md",
                    "--input",
                    "md_dummy.inp",
                ],
                cwd=tmpdir_path,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        template_path = tmpdir_path / "mdrestart"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8", errors="replace")
    return None


def classify_xtb_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if "scf" in lowered and "not converged" in lowered:
        return "fail_scf"
    if "segmentation fault" in lowered:
        return "fail_crash"
    return "fail_other"


def _looks_like_numeric_row(line: str) -> bool:
    parts = line.split()
    if len(parts) < 6:
        return False
    try:
        [float(token.replace("D", "E").replace("d", "e")) for token in parts[:6]]
    except ValueError:
        return False
    return True
