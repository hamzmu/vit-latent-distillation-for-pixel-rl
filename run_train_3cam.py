#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "KMP_USE_SHM": "0",
    "KMP_AFFINITY": "disabled",
    "OMP_WAIT_POLICY": "PASSIVE",
    "MUJOCO_GL": "egl",
    "EGL_PLATFORM": "surfaceless",
    "PYOPENGL_PLATFORM": "egl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Short launcher for 3-camera VTMAE training with sane env defaults."
    )
    p.add_argument("--camera_names", nargs=3, default=["gripperPOV", "corner", "corner2"])
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--schedule", type=str, default="default_3cam")
    p.add_argument("--out", type=str, default="vtmae_3cam_full.pt")
    p.add_argument("--log_file", type=str, default="train3cam.log")

    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="vtmae-3cam")
    p.add_argument("--wandb_run", type=str, default="singlecam_to_allcams")
    p.add_argument("--use_alignment_loss", action="store_true", default=True)
    p.add_argument("--no_alignment_loss", dest="use_alignment_loss", action="store_false")

    p.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Any extra args to pass through to pretrain_vtmae.py",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def build_cmd(args: argparse.Namespace, script_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--camera_names",
        *args.camera_names,
        "--frame_stack",
        str(args.frame_stack),
        "--action_repeat",
        str(args.action_repeat),
        "--patch_size",
        str(args.patch_size),
        "--steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--schedule",
        str(args.schedule),
        "--out",
        str(args.out),
    ]

    if args.wandb:
        cmd.extend(
            [
                "--wandb",
                "--wandb_project",
                str(args.wandb_project),
                "--wandb_run",
                str(args.wandb_run),
            ]
        )

    if not args.use_alignment_loss:
        cmd.append("--no_alignment_loss")

    if args.extra_args:
        cmd.extend(args.extra_args)

    return cmd


def stream_with_tee(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n=== run_train_3cam.py launch ===\n")
        f.write("Command:\n")
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)

        proc.wait()
        f.write(f"\nExit code: {proc.returncode}\n")
        return int(proc.returncode)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    train_script = repo_root / "pretrain_vtmae.py"
    if not train_script.exists():
        print(f"Could not find training script: {train_script}", file=sys.stderr)
        return 2

    cmd = build_cmd(args, train_script)
    log_path = repo_root / args.log_file

    env = os.environ.copy()
    env.update(DEFAULT_ENV)

    print("Launching 3-camera training with:")
    print(" ".join(shlex.quote(c) for c in cmd))
    print(f"Log file: {log_path}")

    if args.dry_run:
        return 0

    return stream_with_tee(cmd, env=env, log_path=log_path)


if __name__ == "__main__":
    raise SystemExit(main())
