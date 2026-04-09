from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.pop("DISPLAY", None)

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import wandb

from metaworld_dm_env import make_metaworld
from pretrain_models import MultiViewVST, MultiViewVSMAE
from pretrain_vtmae import (
    collect_batch,
    composite_recon_from_debug,
    count_params,
    fmt,
    peek_channels,
    pixel_obs_key,
    save_preview,
    set_seed,
)


def _latent_distance(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cosine":
        return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()
    if mode == "mse":
        return F.mse_loss(a, b)
    if mode == "smoothl1":
        return F.smooth_l1_loss(a, b)
    raise AssertionError(f"Unknown latent distance mode: {mode}")


def build_model(
    *,
    image_hw: tuple[int, int],
    patch_size: int,
    dim: int,
    depth: int,
    heads: int,
    mlp_dim: int,
    channels: int,
    decoder_dim: int,
    decoder_depth: int,
    decoder_heads: int,
    num_views: int,
    frame_stack: int,
) -> MultiViewVSMAE:
    encoder = MultiViewVST(
        image_size=image_hw,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        num_views=num_views,
        frame_stack=frame_stack,
    )
    return MultiViewVSMAE(
        encoder=encoder,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        use_sincosmod_encodings=True,
        frame_stack=frame_stack,
        num_views=num_views,
    )


def split_encoder_decoder_params(mae: MultiViewVSMAE) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    encoder_params = list(mae.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in mae.parameters() if id(p) not in encoder_param_ids]
    return encoder_params, decoder_params


def _dedupe_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen = set()
    out = []
    for param in params:
        pid = id(param)
        if pid not in seen:
            out.append(param)
            seen.add(pid)
    return out


def stage2_encoder_side_params(mae: MultiViewVSMAE) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    encoder_params = _dedupe_params(
        list(mae.encoder.parameters())
        + list(mae.encoder_camera_embedding.parameters())
        + list(mae.repr_ln.parameters())
    )
    decoder_params = _dedupe_params(
        list(mae.enc_to_dec.parameters())
        + [mae.mask_token]
        + list(mae.decoder.parameters())
        + list(mae.to_view_pixels.parameters())
        + list(mae.decoder_camera_embedding.parameters())
    )

    all_param_ids = {id(p) for p in mae.parameters()}
    encoder_param_ids = {id(p) for p in encoder_params}
    decoder_param_ids = {id(p) for p in decoder_params}
    assert encoder_param_ids.isdisjoint(decoder_param_ids), "Stage2 encoder/decoder params overlap."
    assert encoder_param_ids | decoder_param_ids == all_param_ids, "Stage2 param split missed some parameters."
    return encoder_params, decoder_params


def configure_stage2_student(student: MultiViewVSMAE, freeze_stage2_decoder: bool) -> None:
    _, decoder_params = stage2_encoder_side_params(student)
    for param in decoder_params:
        param.requires_grad = not freeze_stage2_decoder


def full_visible(batch_size: int, num_views: int, device: torch.device) -> torch.Tensor:
    return torch.ones((batch_size, num_views), dtype=torch.bool, device=device)


def single_visible(batch_size: int, num_views: int, cam_idx: int, device: torch.device) -> torch.Tensor:
    visible = torch.zeros((batch_size, num_views), dtype=torch.bool, device=device)
    visible[:, cam_idx] = True
    return visible


def full_mask_ratios(
    batch_size: int,
    num_views: int,
    mask_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.full((batch_size, num_views), float(mask_ratio), device=device, dtype=dtype)


def single_recon_mask_ratios(
    batch_size: int,
    num_views: int,
    cam_idx: int,
    visible_mask_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ratios = torch.ones((batch_size, num_views), device=device, dtype=dtype)
    ratios[:, cam_idx] = float(visible_mask_ratio)
    return ratios


def _camera_subset_name(cam_idx: int) -> str:
    return f"cam{cam_idx}"


def _default_stage1_out(final_out: str) -> str:
    p = Path(final_out)
    suffix = p.suffix if p.suffix else ".pt"
    stem = p.stem if p.suffix else p.name
    return str(p.with_name(f"{stem}_stage1_teacher{suffix}"))


def _maybe_wandb_log(data: Dict[str, float], step: int, enabled: bool):
    if enabled:
        wandb.log(data, step=step)


def run_stage1(
    *,
    model: MultiViewVSMAE,
    env,
    view_keys: List[str],
    device: torch.device,
    args: argparse.Namespace,
    camera_names: List[str],
    out_images_dir: str,
) -> None:
    model.train()
    encoder_params, decoder_params = split_encoder_decoder_params(model)
    encoder_opt = torch.optim.AdamW(encoder_params, lr=args.stage1_encoder_lr, weight_decay=0.05)
    decoder_opt = torch.optim.AdamW(decoder_params, lr=args.stage1_decoder_lr, weight_decay=0.0)
    running = None

    for step in range(1, args.stage1_steps + 1):
        views = collect_batch(env, args.batch_size, view_keys, device)
        visible_views = full_visible(args.batch_size, len(camera_names), device)
        mask_ratios = full_mask_ratios(
            args.batch_size,
            len(camera_names),
            args.stage1_visible_mask_ratio,
            device,
            views.dtype,
        )
        need_debug = bool(args.vis_every and (step % args.vis_every == 0))

        out = model(
            views,
            visible_views=visible_views,
            mask_ratios=mask_ratios,
            cross_view_loss_weight=args.cross_view_loss_weight,
            return_breakdown=True,
            return_debug=need_debug,
        )
        loss = out["total"]

        encoder_opt.zero_grad(set_to_none=True)
        decoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        encoder_opt.step()
        decoder_opt.step()

        val_total = float(loss.detach().cpu().item())
        running = val_total if running is None else (0.99 * running + 0.01 * val_total)

        if step % args.log_every == 0:
            view_mse = out["per_view_mse"].detach().cpu().tolist()
            print(
                f"[Stage1 {step}/{args.stage1_steps}] total={val_total:.4f} ema={running:.4f} "
                f"v0={view_mse[0]:.4f} v1={view_mse[1]:.4f} v2={view_mse[2]:.4f}",
                flush=True,
            )
            _maybe_wandb_log(
                {
                    "stage1/loss_total": val_total,
                    "stage1/loss_view0": float(out["view0_mse"].detach().cpu().item()),
                    "stage1/loss_view1": float(out["view1_mse"].detach().cpu().item()),
                    "stage1/loss_view2": float(out["view2_mse"].detach().cpu().item()),
                },
                step=step,
                enabled=args.wandb,
            )

        if args.save_every and (step % args.save_every == 0):
            torch.save(model.state_dict(), args.stage1_out)

        if need_debug and "debug" in out:
            recon_views = composite_recon_from_debug(model, views, out["debug"])
            save_preview(
                step=step,
                out_dir=out_images_dir,
                views=views,
                recon_views=recon_views,
                loss_mask=out["debug"]["loss_mask"],
                visible_views=out["debug"]["visible_views"],
                camera_names=camera_names,
                patch_h=model.patch_height,
                patch_w=model.patch_width,
                frame_rgb=3,
            )

    torch.save(model.state_dict(), args.stage1_out)
    print(f"[Stage1] saved teacher weights to {args.stage1_out}", flush=True)


def run_stage2(
    *,
    teacher: MultiViewVSMAE,
    student: MultiViewVSMAE,
    env,
    view_keys: List[str],
    device: torch.device,
    args: argparse.Namespace,
    camera_names: List[str],
    out_images_dir: str,
) -> None:
    teacher.eval()
    student.train()

    student_encoder_params, student_decoder_params = stage2_encoder_side_params(student)
    if args.freeze_stage2_decoder:
        for param in student_decoder_params:
            param.requires_grad = False
        student_decoder_params = []

    encoder_opt = torch.optim.AdamW(student_encoder_params, lr=args.stage2_encoder_lr, weight_decay=0.05)
    decoder_opt = (
        torch.optim.AdamW(student_decoder_params, lr=args.stage2_decoder_lr, weight_decay=0.0)
        if student_decoder_params
        else None
    )
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    running = None

    for step in range(1, args.stage2_steps + 1):
        views = collect_batch(env, args.batch_size, view_keys, device)
        all_visible = full_visible(args.batch_size, len(camera_names), device)

        with torch.no_grad():
            z_teacher = teacher.get_embeddings(views, visible_views=all_visible, eval=True)

        recon_terms: List[torch.Tensor] = []
        distill_terms: List[torch.Tensor] = []
        recon_logs: Dict[str, float] = {}
        distill_logs: Dict[str, float] = {}
        preview_debug = None
        preview_visible = None

        for cam_idx in range(len(camera_names)):
            visible_views = single_visible(args.batch_size, len(camera_names), cam_idx, device)
            mask_ratios = single_recon_mask_ratios(
                args.batch_size,
                len(camera_names),
                cam_idx,
                args.stage2_single_visible_mask_ratio,
                device,
                views.dtype,
            )
            need_debug = bool(
                args.use_stage2_recon_loss
                and args.vis_every
                and (step % args.vis_every == 0)
                and cam_idx == args.preview_cam_idx
            )

            out = None
            if args.use_stage2_recon_loss:
                out = student(
                    views,
                    visible_views=visible_views,
                    mask_ratios=mask_ratios,
                    cross_view_loss_weight=args.cross_view_loss_weight,
                    return_breakdown=True,
                    return_debug=need_debug,
                )
                recon_terms.append(out["total"])
                recon_logs[_camera_subset_name(cam_idx)] = float(out["total"].detach().cpu().item())

            z_single = student.get_embeddings(views, visible_views=visible_views, eval=False)
            distill = _latent_distance(z_single, z_teacher, args.distill_mode)

            distill_terms.append(distill)
            distill_logs[_camera_subset_name(cam_idx)] = float(distill.detach().cpu().item())

            if out is not None and need_debug and "debug" in out:
                preview_debug = out["debug"]
                preview_visible = visible_views

        if recon_terms:
            recon_loss = torch.stack(recon_terms).mean()
        else:
            recon_loss = torch.tensor(0.0, device=device)
        distill_loss = torch.stack(distill_terms).mean()
        total_loss = (
            (float(args.stage2_recon_weight) * recon_loss if args.use_stage2_recon_loss else 0.0)
            + float(args.stage2_distill_weight) * distill_loss
        )

        encoder_opt.zero_grad(set_to_none=True)
        if decoder_opt is not None:
            decoder_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        encoder_opt.step()
        if decoder_opt is not None:
            decoder_opt.step()

        val_total = float(total_loss.detach().cpu().item())
        val_recon = float(recon_loss.detach().cpu().item())
        val_distill = float(distill_loss.detach().cpu().item())
        running = val_total if running is None else (0.99 * running + 0.01 * val_total)

        if step % args.log_every == 0:
            recon_msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(recon_logs.items()))
            distill_msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(distill_logs.items()))
            print(
                f"[Stage2 {step}/{args.stage2_steps}] total={val_total:.4f} ema={running:.4f} "
                f"recon={val_recon:.4f} distill={val_distill:.4f} recon_enabled={int(args.use_stage2_recon_loss)} "
                f"recon_by_cam[{recon_msg}] distill_by_cam[{distill_msg}]",
                flush=True,
            )
            log_data = {
                "stage2/loss_total": val_total,
                "stage2/loss_recon": val_recon,
                "stage2/loss_distill": val_distill,
                "stage2/use_recon_loss": int(args.use_stage2_recon_loss),
                "stage2/freeze_decoder": int(args.freeze_stage2_decoder),
            }
            for key, value in recon_logs.items():
                log_data[f"stage2/recon_{key}"] = value
            for key, value in distill_logs.items():
                log_data[f"stage2/distill_{key}"] = value
            _maybe_wandb_log(log_data, step=step, enabled=args.wandb)

        if args.save_every and (step % args.save_every == 0):
            torch.save(student.state_dict(), args.out)

        if preview_debug is not None and preview_visible is not None:
            recon_views = composite_recon_from_debug(student, views, preview_debug)
            save_preview(
                step=step,
                out_dir=out_images_dir,
                views=views,
                recon_views=recon_views,
                loss_mask=preview_debug["loss_mask"],
                visible_views=preview_visible,
                camera_names=camera_names,
                patch_h=student.patch_height,
                patch_w=student.patch_width,
                frame_rgb=3,
            )

    torch.save(student.state_dict(), args.out)
    print(f"[Stage2] saved student weights to {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Two-stage full-view pretraining plus single-view latent distillation.")
    p.add_argument("--teacher_checkpoint", type=str, default=None)
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env_name", type=str, default="button-press-topdown-v3")
    p.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["gripperPOV", "corner", "corner2"],
        help="Three camera names or ids, e.g. --camera_names gripperPOV corner corner2",
    )

    # Model config. Must match teacher weights if --teacher_checkpoint is used.
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--decoder_dim", type=int, default=256)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--decoder_heads", type=int, default=4)

    # Shared train settings.
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1_000)
    p.add_argument("--vis_every", type=int, default=10_000)
    p.add_argument("--out", type=str, default="vtmae_3cam_distill.pt")
    p.add_argument("--stage1_out", type=str, default=None)
    p.add_argument("--stage1_preview_dir", type=str, default=None)
    p.add_argument("--stage2_preview_dir", type=str, default=None)
    p.add_argument("--preview_cam_idx", type=int, default=0, choices=[0, 1, 2])

    # Stage 1.
    p.add_argument("--stage1_steps", type=int, default=50_000)
    p.add_argument("--stage1_encoder_lr", type=float, default=1e-4)
    p.add_argument("--stage1_decoder_lr", type=float, default=3e-4)
    p.add_argument("--stage1_visible_mask_ratio", type=float, default=0.75)

    # Stage 2.
    p.add_argument("--stage2_steps", type=int, default=50_000)
    p.add_argument("--stage2_encoder_lr", type=float, default=1e-4)
    p.add_argument("--stage2_decoder_lr", type=float, default=3e-4)
    p.add_argument("--stage2_single_visible_mask_ratio", type=float, default=0.75)
    p.add_argument("--use_stage2_recon_loss", dest="use_stage2_recon_loss", action="store_true", default=True)
    p.add_argument("--no_stage2_recon_loss", dest="use_stage2_recon_loss", action="store_false")
    p.add_argument("--freeze_stage2_decoder", dest="freeze_stage2_decoder", action="store_true", default=True)
    p.add_argument("--no_freeze_stage2_decoder", dest="freeze_stage2_decoder", action="store_false")
    p.add_argument("--stage2_recon_weight", type=float, default=1.0)
    p.add_argument("--stage2_distill_weight", type=float, default=1.0)
    p.add_argument("--distill_mode", type=str, default="cosine", choices=["cosine", "mse", "smoothl1"])
    p.add_argument("--cross_view_loss_weight", type=float, default=1.5)

    # WandB.
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vtmae-3cam-distill")
    p.add_argument("--wandb_run", type=str, default=None)

    args = p.parse_args()

    camera_map = {
        "0": "topview",
        "1": "corner",
        "2": "corner2",
        "3": "corner3",
        "4": "behindGripper",
        "5": "gripperPOV",
    }
    camera_names = [camera_map.get(str(c), str(c)) for c in args.camera_names]
    assert len(camera_names) == 3, f"Expected exactly 3 cameras, got {camera_names}"
    assert args.stage1_steps > 0 or args.teacher_checkpoint, (
        "Need either --stage1_steps > 0 or an existing --teacher_checkpoint."
    )
    assert args.stage2_steps > 0, "--stage2_steps must be > 0."

    if args.stage1_out is None:
        args.stage1_out = _default_stage1_out(args.out)
    stage1_preview_dir = args.stage1_preview_dir or f"stage1_{'_'.join(camera_names)}_mw"
    stage2_preview_dir = args.stage2_preview_dir or f"stage2_{Path(args.out).stem}_{'_'.join(camera_names)}_mw"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_metaworld(
        name=args.env_name,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        camera_names=camera_names,
        add_aux_pixels_to_obs=True,
    )
    view_keys = [pixel_obs_key(i) for i in range(len(camera_names))]
    channels, H, W = peek_channels(env, view_keys)
    assert H % args.patch_size == 0 and W % args.patch_size == 0, (
        f"({H},{W}) must be divisible by patch_size={args.patch_size}"
    )

    print(f"[DISTILL-3CAM] channels={channels}, views={len(camera_names)}, size=({H},{W})", flush=True)
    print(f"[DISTILL-3CAM] cameras={camera_names}", flush=True)
    print(
        f"[DISTILL-3CAM] stage1_steps={args.stage1_steps} stage2_steps={args.stage2_steps} "
        f"teacher_checkpoint={args.teacher_checkpoint} freeze_stage2_decoder={int(args.freeze_stage2_decoder)}",
        flush=True,
    )

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    if args.teacher_checkpoint:
        teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
        print(f"[Stage1] skipped, loading teacher from {args.teacher_checkpoint}", flush=True)
    else:
        parent = build_model(
            image_hw=(H, W),
            patch_size=args.patch_size,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            channels=channels,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads,
            num_views=len(camera_names),
            frame_stack=args.frame_stack,
        ).to(device)
        print(
            f"[Stage1] parent total={fmt(count_params(parent))} "
            f"trainable={fmt(count_params(parent, True))}",
            flush=True,
        )
        run_stage1(
            model=parent,
            env=env,
            view_keys=view_keys,
            device=device,
            args=args,
            camera_names=camera_names,
            out_images_dir=stage1_preview_dir,
        )
        teacher_state = parent.state_dict()

    teacher = build_model(
        image_hw=(H, W),
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        channels=channels,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        num_views=len(camera_names),
        frame_stack=args.frame_stack,
    ).to(device)
    student = build_model(
        image_hw=(H, W),
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        channels=channels,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        num_views=len(camera_names),
        frame_stack=args.frame_stack,
    ).to(device)

    teacher.load_state_dict(teacher_state, strict=True)
    student.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    configure_stage2_student(student, args.freeze_stage2_decoder)

    print(
        f"[Stage2] teacher total={fmt(count_params(teacher))} "
        f"student total={fmt(count_params(student))} "
        f"student trainable={fmt(count_params(student, True))} "
        f"decoder_frozen={int(args.freeze_stage2_decoder)}",
        flush=True,
    )
    run_stage2(
        teacher=teacher,
        student=student,
        env=env,
        view_keys=view_keys,
        device=device,
        args=args,
        camera_names=camera_names,
        out_images_dir=stage2_preview_dir,
    )

    if args.wandb:
        wandb.save(args.out)
        if args.stage1_out:
            wandb.save(args.stage1_out)


if __name__ == "__main__":
    main()


"""
Example two-stage run:

python pretrain_vtmae_distill.py \
  --camera_names gripperPOV corner corner2 \
  --frame_stack 3 \
  --action_repeat 2 \
  --patch_size 6 \
  --stage1_steps 50000 \
  --stage2_steps 50000 \
  --batch_size 32 \
  --out vtmae_3cam_distill.pt \
  --wandb \
  --wandb_project vtmae-3cam-distill \
  --wandb_run full_then_single_distill
"""
