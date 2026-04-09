#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
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
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONUNBUFFERED": "1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sequentially train three unfrozen ViT-latent subset-SAC variants on Meta-World "
            "button-press-topdown, focused on earlier encoder co-adaptation and stronger "
            "subset robustness than the previous sweep."
        )
    )
    p.add_argument("--task", type=str, default="button-press-topdown")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_every_steps", type=int, default=10_000)
    p.add_argument("--num_eval_episodes", type=int, default=20)
    p.add_argument("--save_snapshot", action="store_true", default=False)
    p.add_argument("--save_video", action="store_true", default=False)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument(
        "--vit_checkpoint",
        type=str,
        default="seq_runs2/joint_distill_mse_w1p25_20260404_233618/vtmae_joint_distill_mse_w1p25_50000.pt",
    )
    p.add_argument("--python_exe", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


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

    candidates = [Path(sys.executable).resolve()]
    conda_prefix_raw = os.environ.get("CONDA_PREFIX")
    if conda_prefix_raw:
        conda_prefix = Path(conda_prefix_raw)
        candidates.extend(
            [
                conda_prefix / "bin" / "python",
                conda_prefix / "envs" / "vsmae" / "bin" / "python",
            ]
        )
        if conda_prefix.parent.name == "envs":
            candidates.extend(
                [
                    conda_prefix.parent / "vsmae" / "bin" / "python",
                    conda_prefix.parent.parent / "envs" / "vsmae" / "bin" / "python",
                ]
            )

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)

    for candidate in deduped:
        if _python_has_torch(candidate):
            return candidate

    tried = ", ".join(str(p) for p in deduped) if deduped else "<none>"
    raise RuntimeError(f"Could not find a torch-capable Python interpreter. Tried: {tried}")


def detect_compute_device(training_python: Path) -> tuple[str, bool]:
    probe = (
        "import warnings; warnings.filterwarnings('ignore'); "
        "import torch; "
        "cuda=bool(torch.cuda.is_available()); "
        "count=int(torch.cuda.device_count()); "
        "names=[torch.cuda.get_device_name(i) for i in range(count)] if cuda and count>0 else []; "
        "print(f\"GPU (cuda devices={count}: {', '.join(names)})\" if cuda and count>0 else "
        "\"CPU (CUDA unavailable to torch in training environment)\")"
    )
    result = subprocess.run(
        [str(training_python), "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    summary = result.stdout.strip() or "Unknown (device probe produced no output)"
    return summary, summary.startswith("GPU")


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


def stream_with_tee(cmd: list[str], env: dict[str, str], cwd: Path, master_log: Path, method_log: Path) -> int:
    method_log.parent.mkdir(parents=True, exist_ok=True)
    with master_log.open("a", encoding="utf-8") as master_f, method_log.open("a", encoding="utf-8") as method_f:
        for f in (master_f, method_f):
            f.write("Command:\n")
            f.write(" ".join(shlex.quote(c) for c in cmd) + "\n\n")
            f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
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
    mad_root = Path(__file__).resolve().parent
    training_python = resolve_training_python(args)
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    compute_summary, has_gpu = detect_compute_device(training_python)
    if args.device.startswith("cuda") and not has_gpu:
        print(f"Training python: {training_python}")
        print(f"Compute device: {compute_summary}")
        print(f"Requested device {args.device} is unavailable. CUDA is required for this sequence run.")
        return 1

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.out_dir) if args.out_dir else mad_root / "seq_runs" / f"button_press_seq_{run_tag}"
    master_log = base_dir / "seq_training.log"

    unfrozen_variants = [
        {
            "label": "subset_sac_unfrozen_warmup1k_alpha075",
            "folder": "subset_sac_unfrozen_warmup1k_alpha075",
            "extra_overrides": [
                "vit_finetune_encoder=true",
                "vit_encoder_lr=3e-6",
                "vit_unfreeze_after_steps=1000",
                "vit_anchor_reg_weight=1.0",
                "vit_subset_alignment_reg_weight=0.5",
                "latent_aug_alpha=0.75",
                "vit_subset_mode=full_and_singles",
            ],
        },
        {
            "label": "subset_sac_unfrozen_warmup1k_lr2p5e6",
            "folder": "subset_sac_unfrozen_warmup1k_lr2p5e6",
            "extra_overrides": [
                "vit_finetune_encoder=true",
                "vit_encoder_lr=2.5e-6",
                "vit_unfreeze_after_steps=1000",
                "vit_anchor_reg_weight=1.0",
                "vit_subset_alignment_reg_weight=0.5",
                "latent_aug_alpha=0.8",
                "vit_subset_mode=full_and_singles",
            ],
        },
        {
            "label": "subset_sac_unfrozen_warmup1k_reg1p25_align0p6",
            "folder": "subset_sac_unfrozen_warmup1k_reg1p25_align0p6",
            "extra_overrides": [
                "vit_finetune_encoder=true",
                "vit_encoder_lr=3e-6",
                "vit_unfreeze_after_steps=1000",
                "vit_anchor_reg_weight=1.25",
                "vit_subset_alignment_reg_weight=0.6",
                "latent_aug_alpha=0.8",
                "vit_subset_mode=full_and_singles",
            ],
        },
    ]

    methods = [
        # {
        #     "label": "mad",
        #     "agent": "mad",
        #     "folder": "mad_baseline",
        #     "extra_overrides": [],
        # },
        *[
            {
                "label": variant["label"],
                "agent": "vit_latent_subset_sac",
                "folder": variant["folder"],
                "extra_overrides": list(variant["extra_overrides"]),
            }
            for variant in unfrozen_variants
        ],
    ]

    common_overrides = [
        f"task={args.task}",
        "cameras=[first,third1,third2]",
        "eval_cameras=[[first],[third1],[third2],[first,third1,third2]]",
        "camera_alias_profile=mad",
        "metaworld_backend=ours",
        f"num_train_steps={args.steps}",
        f"seed={args.seed}",
        f"device={args.device}",
        f"batch_size={args.batch_size}",
        f"eval_every_steps={args.eval_every_steps}",
        f"num_eval_episodes={args.num_eval_episodes}",
        f"save_snapshot={'true' if args.save_snapshot else 'false'}",
        f"save_video={'true' if args.save_video else 'false'}",
        "save_final_video_once=false",
        f"use_wandb={'true' if args.use_wandb else 'false'}",
    ]

    print(f"Base output directory: {base_dir}")
    print(f"Training python: {training_python}")
    print(f"Compute device: {compute_summary}")
    print(f"Task: {args.task}")
    print("Camera mapping: first->gripperPOV third1->corner2 third2->corner3")
    print(f"ViT checkpoint: {args.vit_checkpoint}")
    print(f"Batch size (all runs): {args.batch_size}")
    print(f"Eval every steps: {args.eval_every_steps}")
    print("Unfrozen subset-SAC variants:")
    for variant in unfrozen_variants:
        print(f"  - {variant['label']}: {' '.join(variant['extra_overrides'])}")

    commands = []
    for method in methods:
        label = method["label"]
        agent_name = method["agent"]
        run_dir = base_dir / method["folder"]
        cmd = [
            str(training_python),
            str(mad_root / "train.py"),
            f"agent={agent_name}",
            *common_overrides,
            *method["extra_overrides"],
        ]
        if agent_name.startswith("vit_latent_"):
            cmd.append(f"vit_checkpoint={args.vit_checkpoint}")
        commands.append((label, agent_name, run_dir, cmd, run_dir / "logs" / "train.log"))
        print(f"{label} dir: {run_dir}")
        print(f"{label}: {' '.join(shlex.quote(c) for c in cmd)}")

    if args.dry_run:
        return 0

    base_dir.mkdir(parents=True, exist_ok=True)
    for label, agent_name, run_dir, cmd, method_log in commands:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[SEQ] Starting {label}", flush=True)
        append_log(master_log, f"\n=== {label} ({agent_name}) ===")
        code = stream_with_tee(cmd, env, run_dir, master_log, method_log)
        if code != 0:
            print(f"[SEQ] {label} failed with exit code {code}. See {master_log}", file=sys.stderr)
            return code
        print(f"[SEQ] Finished {label}", flush=True)

    print(f"[SEQ] All methods completed. Outputs saved in {base_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
