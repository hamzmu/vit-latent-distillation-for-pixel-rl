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
        description=(
            "Sequentially train the selected MAD-camera joint ViT ablations: "
            "cosine distillation with higher single-view reconstruction weight, plus "
            "two MSE-distillation runs with different single-view reconstruction weights."
        )
    )
    p.add_argument("--camera_names", nargs=3, default=["gripperPOV", "corner2", "corner3"])
    p.add_argument("--alias_names", nargs=3, default=["first", "third1", "third2"])
    p.add_argument("--env_name", type=str, default="button-press-topdown-v3")
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1_000)
    p.add_argument("--vis_every", type=int, default=10_000)
    p.add_argument(
        "--cosine_single_recon_weight",
        type=float,
        default=1.25,
        help="single_recon_weight used by the cosine-distillation control run.",
    )
    p.add_argument(
        "--mse_single_recon_weight_low",
        type=float,
        default=1.25,
        help="lower single_recon_weight used by the MSE-distillation run.",
    )
    p.add_argument(
        "--mse_single_recon_weight_high",
        type=float,
        default=1.5,
        help="higher single_recon_weight used by the MSE-distillation run.",
    )
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--log_file", type=str, default="seq_training2.log")
    p.add_argument("--python_exe", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


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
    except Exception:
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

    tried = ", ".join(str(p) for p in candidates) if candidates else "<none>"
    raise RuntimeError(
        "Could not find a torch-capable Python interpreter for joint training variants. "
        f"Tried: {tried}. Pass --python_exe explicitly."
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


def resolve_base_dir(repo_root: Path, out_dir_arg: str | None) -> Path:
    if out_dir_arg:
        return Path(out_dir_arg)
    return repo_root / "seq_runs2"


def resolve_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _value_tag(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def build_method_commands(
    args: argparse.Namespace, repo_root: Path, base_dir: Path, run_tag: str, training_python: Path
) -> list[dict[str, Any]]:
    py = str(training_python)
    steps = int(args.steps)
    assert steps > 0, "--steps must be > 0."
    assert args.cosine_single_recon_weight > 0.0, "--cosine_single_recon_weight must be > 0."
    assert args.mse_single_recon_weight_low > 0.0, "--mse_single_recon_weight_low must be > 0."
    assert args.mse_single_recon_weight_high > 0.0, "--mse_single_recon_weight_high must be > 0."

    common = [
        "--env_name",
        str(args.env_name),
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
        "--steps",
        str(steps),
    ]

    cosine_weight_tag = _value_tag(args.cosine_single_recon_weight)
    mse_low_weight_tag = _value_tag(args.mse_single_recon_weight_low)
    mse_high_weight_tag = _value_tag(args.mse_single_recon_weight_high)

    cosine_dir = base_dir / f"joint_single_recon_cosine_w{cosine_weight_tag}_{run_tag}"
    mse_low_dir = base_dir / f"joint_distill_mse_w{mse_low_weight_tag}_{run_tag}"
    mse_high_dir = base_dir / f"joint_distill_mse_w{mse_high_weight_tag}_{run_tag}"

    cosine_cmd = [
        py,
        str(repo_root / "pretrain_vtmae_joint_distill.py"),
        *common,
        "--single_recon_weight",
        str(args.cosine_single_recon_weight),
        "--out",
        str(cosine_dir / f"vtmae_joint_single_recon_cosine_w{cosine_weight_tag}_{steps}.pt"),
        "--preview_dir",
        str(cosine_dir / "previews"),
    ]

    mse_low_cmd = [
        py,
        str(repo_root / "pretrain_vtmae_joint_distill.py"),
        *common,
        "--single_recon_weight",
        str(args.mse_single_recon_weight_low),
        "--distill_mode",
        "mse",
        "--out",
        str(mse_low_dir / f"vtmae_joint_distill_mse_w{mse_low_weight_tag}_{steps}.pt"),
        "--preview_dir",
        str(mse_low_dir / "previews"),
    ]

    mse_high_cmd = [
        py,
        str(repo_root / "pretrain_vtmae_joint_distill.py"),
        *common,
        "--single_recon_weight",
        str(args.mse_single_recon_weight_high),
        "--distill_mode",
        "mse",
        "--out",
        str(mse_high_dir / f"vtmae_joint_distill_mse_w{mse_high_weight_tag}_{steps}.pt"),
        "--preview_dir",
        str(mse_high_dir / "previews"),
    ]

    return [
        {
            "name": "joint_single_recon_cosine",
            "cmd": cosine_cmd,
            "dir": cosine_dir,
            "log_path": cosine_dir / "logs" / "train.log",
        },
        {
            "name": "joint_distill_mse_low",
            "cmd": mse_low_cmd,
            "dir": mse_low_dir,
            "log_path": mse_low_dir / "logs" / "train.log",
        },
        {
            "name": "joint_distill_mse_high",
            "cmd": mse_high_cmd,
            "dir": mse_high_dir,
            "log_path": mse_high_dir / "logs" / "train.log",
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


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    base_dir = resolve_base_dir(repo_root, args.out_dir)
    run_tag = resolve_run_tag()
    log_path = base_dir / f"{run_tag}_{args.log_file}"

    for script_name in ("pretrain_vtmae_joint_distill.py",):
        script_path = repo_root / script_name
        if not script_path.exists():
            print(f"Could not find training script: {script_path}", file=sys.stderr)
            return 2

    training_python = resolve_training_python(args)
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    commands = build_method_commands(args, repo_root, base_dir, run_tag, training_python)

    print(f"Base output directory: {base_dir}")
    print(f"Run tag: {run_tag}")
    print(f"Master log file: {log_path}")
    print(f"Training python: {training_python}")
    print(f"Compute device: {detect_compute_device(training_python)}")
    print(
        "MAD alias mapping: "
        f"{args.alias_names[0]}->{args.camera_names[0]} "
        f"{args.alias_names[1]}->{args.camera_names[1]} "
        f"{args.alias_names[2]}->{args.camera_names[2]}"
    )
    print(
        "Ablation weights: "
        f"cosine_single_recon={args.cosine_single_recon_weight} "
        f"mse_single_recon_low={args.mse_single_recon_weight_low} "
        f"mse_single_recon_high={args.mse_single_recon_weight_high}"
    )

    if args.dry_run:
        for method in commands:
            print(f"{method['name']} dir: {method['dir']}")
            print(f"{method['name']} log: {method['log_path']}")
            print(f"{method['name']}: {' '.join(shlex.quote(c) for c in method['cmd'])}")
        return 0

    base_dir.mkdir(parents=True, exist_ok=True)
    for method in commands:
        Path(method["dir"]).mkdir(parents=True, exist_ok=True)
        print(f"{method['name']} dir: {method['dir']}")
        print(f"{method['name']} log: {method['log_path']}")
        print(f"{method['name']}: {' '.join(shlex.quote(c) for c in method['cmd'])}")

    for method in commands:
        method_name = str(method["name"])
        print(f"\n[SEQ2] Starting {method_name}", flush=True)
        code = stream_with_tee(
            method_name=method_name,
            cmd=list(method["cmd"]),
            env=env,
            log_path=log_path,
            method_log_path=Path(method["log_path"]),
        )
        if code != 0:
            print(f"[SEQ2] {method_name} failed with exit code {code}. See {log_path}", file=sys.stderr)
            return code
        print(f"[SEQ2] Finished {method_name}", flush=True)

    print(f"[SEQ2] All joint variants completed. Outputs saved in {base_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
