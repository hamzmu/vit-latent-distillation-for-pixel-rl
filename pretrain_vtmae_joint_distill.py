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


def _dedupe_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen = set()
    out = []
    for param in params:
        pid = id(param)
        if pid not in seen:
            out.append(param)
            seen.add(pid)
    return out


def split_encoder_decoder_params(mae: MultiViewVSMAE) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
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
    assert encoder_param_ids.isdisjoint(decoder_param_ids), "Encoder/decoder param split overlap."
    assert encoder_param_ids | decoder_param_ids == all_param_ids, "Encoder/decoder split missed parameters."
    return encoder_params, decoder_params


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


def _maybe_wandb_log(data: Dict[str, float], step: int, enabled: bool):
    if enabled:
        wandb.log(data, step=step)


def _distill_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))


def run_joint_training(
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
    encoder_opt = torch.optim.AdamW(encoder_params, lr=args.encoder_lr, weight_decay=0.05)
    decoder_opt = torch.optim.AdamW(decoder_params, lr=args.decoder_lr, weight_decay=0.0)
    running = None
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        views = collect_batch(env, args.batch_size, view_keys, device)
        all_visible = full_visible(args.batch_size, len(camera_names), device)
        all_mask_ratios = full_mask_ratios(
            args.batch_size,
            len(camera_names),
            args.full_visible_mask_ratio,
            device,
            views.dtype,
        )

        out_full = model(
            views,
            visible_views=all_visible,
            mask_ratios=all_mask_ratios,
            cross_view_loss_weight=args.cross_view_loss_weight,
            return_breakdown=True,
            return_debug=False,
        )

        with torch.no_grad():
            z_full = model.get_embeddings(views, visible_views=all_visible, eval=False)

        single_recon_terms: List[torch.Tensor] = []
        distill_terms: List[torch.Tensor] = []
        single_recon_logs: Dict[str, float] = {}
        distill_logs: Dict[str, float] = {}
        preview_debug = None
        preview_visible = None

        for cam_idx in range(len(camera_names)):
            visible_views = single_visible(args.batch_size, len(camera_names), cam_idx, device)
            mask_ratios = single_recon_mask_ratios(
                args.batch_size,
                len(camera_names),
                cam_idx,
                args.single_visible_mask_ratio,
                device,
                views.dtype,
            )
            need_debug = bool(
                args.use_single_recon_loss
                and args.vis_every
                and (step % args.vis_every == 0)
                and cam_idx == args.preview_cam_idx
            )

            out_single = None
            if args.use_single_recon_loss:
                out_single = model(
                    views,
                    visible_views=visible_views,
                    mask_ratios=mask_ratios,
                    cross_view_loss_weight=args.cross_view_loss_weight,
                    return_breakdown=True,
                    return_debug=need_debug,
                )
                single_recon_terms.append(out_single["total"])
                single_recon_logs[_camera_subset_name(cam_idx)] = float(out_single["total"].detach().cpu().item())

            z_single = model.get_embeddings(views, visible_views=visible_views, eval=False)
            distill = _latent_distance(z_single, z_full, args.distill_mode)
            distill_terms.append(distill)
            distill_logs[_camera_subset_name(cam_idx)] = float(distill.detach().cpu().item())

            if out_single is not None and need_debug and "debug" in out_single:
                preview_debug = out_single["debug"]
                preview_visible = visible_views

        full_recon_loss = out_full["total"]
        single_recon_loss = (
            torch.stack(single_recon_terms).mean()
            if single_recon_terms
            else torch.tensor(0.0, device=device)
        )
        distill_loss = torch.stack(distill_terms).mean()
        distill_scale = _distill_scale(step, args.distill_warmup_steps)

        total_loss = (
            float(args.full_recon_weight) * full_recon_loss
            + (float(args.single_recon_weight) * single_recon_loss if args.use_single_recon_loss else 0.0)
            + float(args.distill_weight) * distill_scale * distill_loss
        )

        encoder_opt.zero_grad(set_to_none=True)
        decoder_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        encoder_opt.step()
        decoder_opt.step()

        val_total = float(total_loss.detach().cpu().item())
        val_full = float(full_recon_loss.detach().cpu().item())
        val_single = float(single_recon_loss.detach().cpu().item())
        val_distill = float(distill_loss.detach().cpu().item())
        running = val_total if running is None else (0.99 * running + 0.01 * val_total)

        if step % args.log_every == 0:
            full_view_mse = out_full["per_view_mse"].detach().cpu().tolist()
            single_recon_msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(single_recon_logs.items()))
            distill_msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(distill_logs.items()))
            print(
                f"[{step}/{args.steps}] total={val_total:.4f} ema={running:.4f} "
                f"full={val_full:.4f} single={val_single:.4f} distill={val_distill:.4f} "
                f"distill_scale={distill_scale:.3f} single_recon_enabled={int(args.use_single_recon_loss)} "
                f"full_v0={full_view_mse[0]:.4f} full_v1={full_view_mse[1]:.4f} full_v2={full_view_mse[2]:.4f} "
                f"single_by_cam[{single_recon_msg}] distill_by_cam[{distill_msg}]",
                flush=True,
            )
            _maybe_wandb_log(
                {
                    "joint/loss_total": val_total,
                    "joint/loss_full_recon": val_full,
                    "joint/loss_single_recon": val_single,
                    "joint/loss_distill": val_distill,
                    "joint/distill_scale": distill_scale,
                    "joint/use_single_recon_loss": int(args.use_single_recon_loss),
                    "joint/full_view0": float(out_full["view0_mse"].detach().cpu().item()),
                    "joint/full_view1": float(out_full["view1_mse"].detach().cpu().item()),
                    "joint/full_view2": float(out_full["view2_mse"].detach().cpu().item()),
                    **{f"joint/single_{k}": v for k, v in single_recon_logs.items()},
                    **{f"joint/distill_{k}": v for k, v in distill_logs.items()},
                },
                step=step,
                enabled=args.wandb,
            )

        if args.save_every and (step % args.save_every == 0):
            torch.save(model.state_dict(), args.out)

        if preview_debug is not None and preview_visible is not None:
            recon_views = composite_recon_from_debug(model, views, preview_debug)
            save_preview(
                step=step,
                out_dir=out_images_dir,
                views=views,
                recon_views=recon_views,
                loss_mask=preview_debug["loss_mask"],
                visible_views=preview_visible,
                camera_names=camera_names,
                patch_h=model.patch_height,
                patch_w=model.patch_width,
                frame_rgb=3,
            )

    torch.save(model.state_dict(), args.out)
    print(f"Done. Saved weights to {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Joint full-view reconstruction plus same-batch single-view latent distillation."
    )
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

    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--decoder_dim", type=int, default=256)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--decoder_heads", type=int, default=4)

    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--decoder_lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1_000)
    p.add_argument("--vis_every", type=int, default=10_000)
    p.add_argument("--out", type=str, default="vtmae_joint_distill.pt")
    p.add_argument("--preview_dir", type=str, default=None)
    p.add_argument("--preview_cam_idx", type=int, default=0, choices=[0, 1, 2])

    p.add_argument("--full_visible_mask_ratio", type=float, default=0.75)
    p.add_argument("--single_visible_mask_ratio", type=float, default=0.75)
    p.add_argument("--use_single_recon_loss", dest="use_single_recon_loss", action="store_true", default=True)
    p.add_argument("--no_single_recon_loss", dest="use_single_recon_loss", action="store_false")
    p.add_argument("--full_recon_weight", type=float, default=1.0)
    p.add_argument("--single_recon_weight", type=float, default=1.0)
    p.add_argument("--distill_weight", type=float, default=1.0)
    p.add_argument("--distill_warmup_steps", type=int, default=0)
    p.add_argument("--distill_mode", type=str, default="cosine", choices=["cosine", "mse", "smoothl1"])
    p.add_argument("--cross_view_loss_weight", type=float, default=1.5)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vtmae-joint-distill")
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

    print(f"[JOINT-3CAM] channels={channels}, views={len(camera_names)}, size=({H},{W})", flush=True)
    print(f"[JOINT-3CAM] cameras={camera_names}", flush=True)
    print(
        f"[JOINT-3CAM] steps={args.steps} distill_mode={args.distill_mode} "
        f"single_recon={int(args.use_single_recon_loss)} warmup={args.distill_warmup_steps}",
        flush=True,
    )

    model = build_model(
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

    print(f"[Params] total={fmt(count_params(model))} trainable={fmt(count_params(model, True))}", flush=True)
    print(
        "[Weights] "
        f"full_recon={args.full_recon_weight} "
        f"single_recon={args.single_recon_weight} "
        f"distill={args.distill_weight}",
        flush=True,
    )

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    run_joint_training(
        model=model,
        env=env,
        view_keys=view_keys,
        device=device,
        args=args,
        camera_names=camera_names,
        out_images_dir=args.preview_dir or f"joint_{Path(args.out).stem}_{'_'.join(camera_names)}_mw",
    )

    if args.wandb:
        wandb.save(args.out)


if __name__ == "__main__":
    main()


"""
Example run:

python pretrain_vtmae_joint_distill.py \
  --camera_names gripperPOV corner corner2 \
  --frame_stack 3 \
  --action_repeat 2 \
  --patch_size 6 \
  --steps 100000 \
  --batch_size 32 \
  --out vtmae_joint_distill.pt \
  --wandb \
  --wandb_project vtmae-joint-distill \
  --wandb_run joint_full_and_single
"""
