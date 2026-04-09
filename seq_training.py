#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
    "PYTHONUNBUFFERED": "1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequentially train joint distill, two-stage distill, and curriculum 3-camera models."
    )
    p.add_argument("--camera_names", nargs=3, default=["gripperPOV", "corner", "corner2"])
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--steps_per_method", type=int, default=50_000)
    p.add_argument("--distill_stage1_steps", type=int, default=None)
    p.add_argument("--distill_stage2_steps", type=int, default=None)

    p.add_argument("--schedule", type=str, default="default_3cam")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1_000)
    p.add_argument("--vis_every", type=int, default=10_000)

    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--log_file", type=str, default="seq_training.log")
    p.add_argument(
        "--python_exe",
        type=str,
        default=None,
        help="Optional explicit Python executable to use for child training scripts.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Ignored. seq_training always disables wandb so all methods use the same local-only logging path.",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def resolve_base_dir(repo_root: Path, out_dir_arg: str | None) -> Path:
    if out_dir_arg:
        return Path(out_dir_arg)
    return repo_root / "seq_runs"


def resolve_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_distill_steps(args: argparse.Namespace) -> tuple[int, int]:
    if args.distill_stage1_steps is None and args.distill_stage2_steps is None:
        stage1 = args.steps_per_method // 2
        stage2 = args.steps_per_method - stage1
        return stage1, stage2
    if args.distill_stage1_steps is None:
        stage2 = int(args.distill_stage2_steps)
        stage1 = args.steps_per_method - stage2
        return stage1, stage2
    if args.distill_stage2_steps is None:
        stage1 = int(args.distill_stage1_steps)
        stage2 = args.steps_per_method - stage1
        return stage1, stage2
    return int(args.distill_stage1_steps), int(args.distill_stage2_steps)


def _append_candidate(candidates: list[Path], candidate: Path) -> None:
    if candidate.exists() and candidate not in candidates:
        candidates.append(candidate)


def _python_has_torch(python_exe: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python_exe), "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception as exc:
        return False


def resolve_training_python(args: argparse.Namespace) -> Path:
    if args.python_exe:
        python_exe = Path(args.python_exe).expanduser().resolve()
        if not python_exe.exists():
            raise FileNotFoundError(f"--python_exe does not exist: {python_exe}")
        return python_exe

    candidates: list[Path] = []
    current_python = Path(sys.executable).resolve()
    _append_candidate(candidates, current_python)

    conda_prefix_raw = os.environ.get("CONDA_PREFIX")
    if conda_prefix_raw:
        conda_prefix = Path(conda_prefix_raw)
        _append_candidate(candidates, conda_prefix / "bin" / "python")
        if conda_prefix.name == "envs":
            _append_candidate(candidates, conda_prefix / "vsmae" / "bin" / "python")
        else:
            _append_candidate(candidates, conda_prefix / "envs" / "vsmae" / "bin" / "python")
            if conda_prefix.parent.name == "envs":
                _append_candidate(candidates, conda_prefix.parent / "vsmae" / "bin" / "python")
                _append_candidate(candidates, conda_prefix.parent.parent / "envs" / "vsmae" / "bin" / "python")

    for base in (current_python.parent.parent, current_python.parent.parent.parent):
        if base.exists():
            _append_candidate(candidates, base / "envs" / "vsmae" / "bin" / "python")

    for candidate in candidates:
        if _python_has_torch(candidate):
            return candidate

    candidate_list = ", ".join(str(p) for p in candidates) if candidates else "<none>"
    raise RuntimeError(
        "Could not find a torch-capable Python interpreter for child training scripts. "
        f"Tried: {candidate_list}. Pass --python_exe explicitly."
    )


def detect_compute_device(training_python: Path) -> str:
    probe = (
        "import warnings; warnings.filterwarnings('ignore'); "
        "import torch; "
        "cuda=bool(torch.cuda.is_available()); "
        "count=int(torch.cuda.device_count()); "
        "names=[torch.cuda.get_device_name(i) for i in range(count)] if cuda and count>0 else []; "
        "print(f\"GPU (cuda devices={count}: {', '.join(names)})\" if cuda and count>0 else "
        "\"CPU (CUDA unavailable to torch in training environment)\")"
    )
    try:
        result = subprocess.run(
            [str(training_python), "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        status = result.stdout.strip()
        if status:
            return status
        return "Unknown (device probe produced no output)"
    except Exception as exc:
        return f"Unknown (device probe failed: {exc})"


def distill_stage1_out_path(out_dir: Path, steps: int) -> Path:
    return out_dir / f"vtmae_distill_{steps}_stage1_teacher.pt"


def build_method_commands(
    args: argparse.Namespace, repo_root: Path, base_dir: Path, run_tag: str, training_python: Path
) -> list[dict[str, Any]]:
    py = str(training_python)
    steps = int(args.steps_per_method)
    distill_stage1, distill_stage2 = resolve_distill_steps(args)

    assert steps > 0, "--steps_per_method must be > 0."
    assert distill_stage1 > 0 and distill_stage2 > 0, "Distill stage steps must both be > 0."
    assert distill_stage1 + distill_stage2 == steps, (
        f"Distill stage split must sum to steps_per_method={steps}, got {distill_stage1}+{distill_stage2}."
    )

    joint_dir = base_dir / f"joint_{run_tag}"
    curriculum_dir = base_dir / f"curriculum_{run_tag}"
    distill_dir = base_dir / f"distill_{run_tag}"
    joint_log = joint_dir / "logs" / "train.log"
    distill_log = distill_dir / "logs" / "train.log"
    curriculum_log = curriculum_dir / "logs" / "train.log"
    joint_out = joint_dir / f"vtmae_joint_distill_{steps}.pt"
    curriculum_out = curriculum_dir / f"vtmae_curriculum_{steps}.pt"
    distill_stage1_out = distill_dir / f"vtmae_distill_{steps}_stage1_teacher.pt"
    distill_out = distill_dir / f"vtmae_distill_{steps}.pt"
    common = [
        "--camera_names",
        *args.camera_names,
        "--frame_stack",
        str(args.frame_stack),
        "--action_repeat",
        str(args.action_repeat),
        "--patch_size",
        str(args.patch_size),
        "--batch_size",
        str(args.batch_size),
        "--log_every",
        str(args.log_every),
        "--save_every",
        str(args.save_every),
        "--vis_every",
        str(args.vis_every),
    ]

    joint_cmd = [
        py,
        str(repo_root / "pretrain_vtmae_joint_distill.py"),
        *common,
        "--steps",
        str(steps),
        "--out",
        str(joint_out),
        "--preview_dir",
        str(joint_dir / "previews"),
    ]

    curriculum_cmd = [
        py,
        str(repo_root / "pretrain_vtmae.py"),
        *common,
        "--steps",
        str(steps),
        "--schedule",
        str(args.schedule),
        "--out",
        str(curriculum_out),
        "--preview_dir",
        str(curriculum_dir / "previews"),
    ]

    distill_cmd = [
        py,
        str(repo_root / "pretrain_vtmae_distill.py"),
        *common,
        "--stage1_steps",
        str(distill_stage1),
        "--stage2_steps",
        str(distill_stage2),
        "--stage1_out",
        str(distill_stage1_out),
        "--out",
        str(distill_out),
        "--stage1_preview_dir",
        str(distill_dir / "stage1_previews"),
        "--stage2_preview_dir",
        str(distill_dir / "stage2_previews"),
    ]

    return [
        {
            "name": "joint",
            "cmd": joint_cmd,
            "dir": joint_dir,
            "log_path": joint_log,
            "stage1_teacher_path": None,
        },
        {
            "name": "distill",
            "cmd": distill_cmd,
            "dir": distill_dir,
            "log_path": distill_log,
            "stage1_teacher_path": distill_stage1_out,
        },
        {
            "name": "curriculum",
            "cmd": curriculum_cmd,
            "dir": curriculum_dir,
            "log_path": curriculum_log,
            "stage1_teacher_path": None,
        },
    ]


def stream_with_tee(
    *,
    method_name: str,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    method_log_path: Path,
) -> int:
    method_log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as master_f, method_log_path.open("a", encoding="utf-8") as method_f:
        for f in (master_f, method_f):
            f.write(f"\n=== {method_name} ===\n")
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
            sys.stdout.flush()
            master_f.write(line)
            method_f.write(line)
            master_f.flush()
            method_f.flush()

        proc.wait()
        for f in (master_f, method_f):
            f.write(f"\nExit code: {proc.returncode}\n")
            f.flush()
        return int(proc.returncode)


def append_log_message(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    base_dir = resolve_base_dir(repo_root, args.out_dir)
    run_tag = resolve_run_tag()
    log_path = base_dir / f"{run_tag}_{args.log_file}"

    for script_name in ("pretrain_vtmae.py", "pretrain_vtmae_distill.py", "pretrain_vtmae_joint_distill.py"):
        script_path = repo_root / script_name
        if not script_path.exists():
            print(f"Could not find training script: {script_path}", file=sys.stderr)
            return 2

    training_python = resolve_training_python(args)
    commands = build_method_commands(args, repo_root, base_dir, run_tag, training_python)
    env = os.environ.copy()
    env.update(DEFAULT_ENV)

    distill_stage1, distill_stage2 = resolve_distill_steps(args)
    print(f"Base output directory: {base_dir}")
    print(f"Run tag: {run_tag}")
    print(f"Master log file: {log_path}")
    print(f"Training python: {training_python}")
    print(f"Compute device: {detect_compute_device(training_python)}")
    print(f"WandB logging: disabled in seq_training (local text logs only)")
    print(
        f"Per-method steps: joint={args.steps_per_method} "
        f"distill={distill_stage1}+{distill_stage2} "
        f"curriculum={args.steps_per_method}"
    )
    if args.dry_run:
        for method in commands:
            print(f"{method['name']} dir: {method['dir']}")
            print(f"{method['name']} log: {method['log_path']}")
            print(f"{method['name']}: {' '.join(shlex.quote(c) for c in method['cmd'])}")
        return 0

    base_dir.mkdir(parents=True, exist_ok=True)
    for method in commands:
        method["dir"].mkdir(parents=True, exist_ok=True)
        print(f"{method['name']} dir: {method['dir']}")
        print(f"{method['name']} log: {method['log_path']}")
        print(f"{method['name']}: {' '.join(shlex.quote(c) for c in method['cmd'])}")

    for method in commands:
        method_name = str(method["name"])
        cmd = list(method["cmd"])
        print(f"\n[SEQ] Starting {method_name}", flush=True)
        code = stream_with_tee(
            method_name=method_name,
            cmd=cmd,
            env=env,
            log_path=log_path,
            method_log_path=Path(method["log_path"]),
        )
        if code != 0:
            print(f"[SEQ] {method_name} failed with exit code {code}. See {log_path}", file=sys.stderr)
            return code
        if method_name == "distill":
            teacher_ckpt = Path(method["stage1_teacher_path"])
            if teacher_ckpt.exists():
                teacher_ckpt.unlink()
                msg = f"[SEQ] Removed temporary distill teacher checkpoint: {teacher_ckpt}"
                print(msg, flush=True)
                append_log_message(log_path, msg)
        print(f"[SEQ] Finished {method_name}", flush=True)

    print(f"[SEQ] All methods completed. Outputs saved in {base_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
