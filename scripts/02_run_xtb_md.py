#!/usr/bin/env python
"""Prepare and execute batch xTB Born-Oppenheimer MD trajectories."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import (
    ensure_directory,
    read_json,
    read_last_xyz_frame,
    read_xyz,
    write_json,
    write_xyz,
)
from utils.xtb import (
    build_mdrestart_text,
    classify_xtb_failure,
    load_md_config,
    maybe_generate_mdrestart_template,
    write_md_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/md_default.yaml", help="MD configuration YAML")
    parser.add_argument("--samples-dir", default=None, help="Override samples directory")
    parser.add_argument("--runs-dir", default=None, help="Override runs directory")
    parser.add_argument("--workers", type=int, default=None, help="Override parallel worker count")
    parser.add_argument("--shard", default=None, help="Shard spec of the form I/N, zero-based index")
    parser.add_argument("--xtb-bin", default="xtb", help="xTB executable")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare xTB inputs without launching xTB")
    parser.add_argument("--force", action="store_true", help="Re-run trajectories even if status.json exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_md_config(args.config)
    if args.samples_dir:
        config["samples_dir"] = args.samples_dir
    if args.runs_dir:
        config["runs_dir"] = args.runs_dir
    if args.workers is not None:
        config["parallel"]["workers"] = args.workers

    samples_dir = Path(config["samples_dir"])
    runs_dir = ensure_directory(config["runs_dir"])
    sample_dirs = sorted(path for path in samples_dir.glob("traj_*") if path.is_dir())
    sample_dirs = apply_shard(sample_dirs, args.shard)

    jobs = []
    for sample_dir in sample_dirs:
        run_dir = runs_dir / sample_dir.name
        status_path = run_dir / "status.json"
        if status_path.exists() and not args.force:
            status = read_json(status_path)
            if status.get("status") == "success":
                continue
            if status.get("status") == "dry_run_prepared" and args.dry_run:
                continue
        jobs.append(
            {
                "sample_dir": str(sample_dir),
                "run_dir": str(run_dir),
                "config": deepcopy(config),
                "xtb_bin": args.xtb_bin,
                "dry_run": args.dry_run,
            }
        )

    if not jobs:
        print("No trajectories to run.")
        return

    workers = max(1, int(config["parallel"]["workers"]))
    results = []
    if workers == 1:
        for job in jobs:
            results.append(run_single_trajectory(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_single_trajectory, job): job for job in jobs}
            for future in as_completed(future_map):
                results.append(future.result())

    success = sum(result["status"] == "success" for result in results)
    prepared = sum(result["status"] == "dry_run_prepared" for result in results)
    print(f"Processed {len(results)} trajectories: success={success}, dry_run_prepared={prepared}")
    for result in results:
        print(f"{result['traj_id']}: {result['status']}")


def apply_shard(sample_dirs: list[Path], shard: str | None) -> list[Path]:
    if shard is None:
        return sample_dirs
    left, right = shard.split("/")
    shard_idx = int(left)
    shard_total = int(right)
    if shard_idx < 0 or shard_idx >= shard_total:
        raise ValueError(f"Invalid shard spec {shard}")
    return [path for idx, path in enumerate(sample_dirs) if idx % shard_total == shard_idx]


def run_single_trajectory(job: dict) -> dict:
    sample_dir = Path(job["sample_dir"])
    run_dir = ensure_directory(job["run_dir"])
    config = deepcopy(job["config"])
    xtb_bin = job["xtb_bin"]
    dry_run = job["dry_run"]

    traj_id = sample_dir.name
    started_at = time.time()
    status_payload = {
        "traj_id": traj_id,
        "sample_dir": str(sample_dir.as_posix()),
        "run_dir": str(run_dir.as_posix()),
        "started_at_epoch_s": started_at,
    }

    try:
        symbols, coords_ang, comment = read_xyz(sample_dir / "init.xyz")
        velocities_ang_fs = np.load(sample_dir / "init.vel.npy")
        if velocities_ang_fs.shape != coords_ang.shape:
            raise ValueError("Velocity array shape does not match the coordinate array.")

        with (run_dir / "md.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        write_xyz(run_dir / "coord.xyz", symbols, coords_ang, comment=comment)
        write_md_input(run_dir / "md.inp", config)

        template_text = None
        if not dry_run:
            template_text = maybe_generate_mdrestart_template(
                xtb_bin=xtb_bin,
                coord_xyz=run_dir / "coord.xyz",
                config=config,
                charge=int(config["charge"]),
                uhf=int(config["uhf"]),
                gfn=int(config["gfn"]),
            )
        mdrestart_text = build_mdrestart_text(coords_ang, velocities_ang_fs, template_text=template_text)
        (run_dir / "mdrestart").write_text(mdrestart_text, encoding="utf-8")

        if dry_run:
            status_payload["status"] = "dry_run_prepared"
            status_payload["finished_at_epoch_s"] = time.time()
            write_json(run_dir / "status.json", status_payload)
            return status_payload

        if shutil.which(xtb_bin) is None:
            raise FileNotFoundError(f"xTB executable '{xtb_bin}' was not found on PATH.")

        md_command = [
            xtb_bin,
            "coord.xyz",
            "--chrg",
            str(config["charge"]),
            "--uhf",
            str(config["uhf"]),
            "--gfn",
            str(config["gfn"]),
            "--md",
            "--input",
            "md.inp",
        ]
        timeout_s = int(config["parallel"]["timeout_per_traj_s"])
        with (run_dir / "xtb.log").open("w", encoding="utf-8") as handle:
            try:
                subprocess.run(
                    md_command,
                    cwd=run_dir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                status_payload["status"] = "fail_timeout"
                status_payload["finished_at_epoch_s"] = time.time()
                write_json(run_dir / "status.json", status_payload)
                return status_payload

        log_text = (run_dir / "xtb.log").read_text(encoding="utf-8", errors="replace")
        if not (run_dir / "xtbmdok").exists():
            status_payload["status"] = classify_xtb_failure(log_text)
            status_payload["finished_at_epoch_s"] = time.time()
            write_json(run_dir / "status.json", status_payload)
            return status_payload

        final_sp_dir = ensure_directory(run_dir / "final_sp")
        symbols_last, coords_last, comment_last = read_last_xyz_frame(run_dir / "xtb.trj")
        write_xyz(final_sp_dir / "final.xyz", symbols_last, coords_last, comment=comment_last)

        with (final_sp_dir / "sp.log").open("w", encoding="utf-8") as handle:
            subprocess.run(
                [
                    xtb_bin,
                    "final.xyz",
                    "--chrg",
                    str(config["charge"]),
                    "--uhf",
                    str(config["uhf"]),
                    "--gfn",
                    str(config["gfn"]),
                ],
                cwd=final_sp_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=max(600, timeout_s // 4),
                check=False,
            )

        charges_path = final_sp_dir / "charges"
        status_payload["status"] = "success" if charges_path.exists() else "fail_other"
        status_payload["finished_at_epoch_s"] = time.time()
        status_payload["duration_s"] = status_payload["finished_at_epoch_s"] - started_at
        write_json(run_dir / "status.json", status_payload)
        return status_payload

    except Exception as exc:
        status_payload["status"] = "fail_other"
        status_payload["error"] = str(exc)
        status_payload["finished_at_epoch_s"] = time.time()
        write_json(run_dir / "status.json", status_payload)
        return status_payload


if __name__ == "__main__":
    main()
