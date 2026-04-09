from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.pop("DISPLAY", None)

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from einops import rearrange

from metaworld_dm_env import make_metaworld
from pretrain_models import MultiViewVST, MultiViewVSMAE


# --------------------------- utils ---------------------------

def set_seed(seed: int):
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def fmt(n: int) -> str:
    return f"{n:,}"


def pixel_obs_key(view_idx: int) -> str:
    if view_idx == 0:
        return "pixels"
    if view_idx == 1:
        return "pixels_aux"
    return f"pixels_aux{view_idx}"


@torch.no_grad()
def _get_obs_dict(ts, view_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    obs = ts.observation if hasattr(ts, "observation") else ts
    missing = [k for k in view_keys if k not in obs]
    assert not missing, f"Missing view keys {missing}; available={list(obs.keys())}"
    return {k: obs[k] for k in view_keys}


def _to_chw_stacked(obs_item: np.ndarray) -> torch.Tensor:
    """
    Accepts either:
      - [S, H, W, C]  -> returns [C*S, H, W]
      - [H, W, C]     -> returns [C, H, W]
      - [C, H, W]     -> returns as-is
    """
    if obs_item.ndim == 4:  # [S,H,W,C]
        S, H, W, C = obs_item.shape
        x = torch.from_numpy(obs_item).permute(0, 3, 1, 2).contiguous()  # [S,C,H,W]
        return x.view(S * C, H, W)
    if obs_item.ndim == 3:
        if obs_item.shape[-1] in (1, 3, 4):
            return torch.from_numpy(obs_item).permute(2, 0, 1).contiguous()
        return torch.from_numpy(obs_item)
    raise AssertionError(f"unexpected obs shape {obs_item.shape}")


@torch.no_grad()
def _sample_action(env) -> np.ndarray:
    spec = env.action_spec()
    shape = spec.shape if getattr(spec, "shape", ()) else (1,)
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def peek_channels(env, view_keys: Sequence[str]) -> Tuple[int, int, int]:
    ts = env.reset()
    obs = _get_obs_dict(ts, view_keys)

    channels = []
    base_hw = None
    for key in view_keys:
        chw = _to_chw_stacked(obs[key])
        c, h, w = chw.shape
        channels.append(c)
        if base_hw is None:
            base_hw = (h, w)
        else:
            assert (h, w) == base_hw, f"Spatial mismatch for key {key}: {(h, w)} vs {base_hw}"

    assert len(set(channels)) == 1, f"All views must have same channels, got {channels}"
    return channels[0], base_hw[0], base_hw[1]


def collect_batch(env, batch_size: int, view_keys: Sequence[str], device: torch.device) -> torch.Tensor:
    """
    Returns views tensor [B, V, C, H, W] normalized to [0,1].
    """
    batch_views: List[torch.Tensor] = []

    ts = env.reset()
    for _ in range(batch_size):
        obs = _get_obs_dict(ts, view_keys)
        per_view = [_to_chw_stacked(obs[k]) for k in view_keys]  # list of [C,H,W]
        batch_views.append(torch.stack(per_view, dim=0))

        ts = env.step(_sample_action(env))
        if hasattr(ts, "last") and ts.last():
            ts = env.reset()

    views = torch.stack(batch_views, dim=0).float().to(device) / 255.0
    return views


# --------------------------- patch helpers ---------------------------

def unpatchify(patches: torch.Tensor, C: int, H: int, W: int, ph: int, pw: int) -> torch.Tensor:
    """
    patches: [B, N, ph*pw*C] -> [B, C, H, W]
    """
    h = H // ph
    w = W // pw
    return rearrange(patches, "b (h w) (ph pw c) -> b c (h ph) (w pw)", h=h, w=w, ph=ph, pw=pw, c=C)


@torch.no_grad()
def composite_recon_from_debug(mae: MultiViewVSMAE, views: torch.Tensor, debug: dict) -> torch.Tensor:
    """
    Uses model debug outputs to build composite reconstructions:
      - unmasked patches copied from input
      - masked patches replaced with decoder predictions

    Returns [B, V, C, H, W]
    """
    target_patches = debug["target_patches"]  # [B,V,N,P]
    pred_patches = debug["pred_patches"]      # [B,V,N,P]
    loss_mask = debug["loss_mask"]            # [B,V,N]

    recon_patches = target_patches.clone()
    recon_patches[loss_mask] = pred_patches[loss_mask]

    B, V, _, _ = recon_patches.shape
    _, _, C, H, W = views.shape
    ph, pw = mae.patch_height, mae.patch_width

    recon_views = []
    for vidx in range(V):
        recon_v = unpatchify(recon_patches[:, vidx], C=C, H=H, W=W, ph=ph, pw=pw)
        recon_views.append(recon_v)
    return torch.stack(recon_views, dim=1).clamp(0, 1)


# --------------------------- visualization helpers ---------------------------

def _draw_patch_boxes_single(
    img: torch.Tensor,
    masked_indices: torch.Tensor,
    ph: int,
    pw: int,
    total_patches: int,
    color=(0.0, 0.0, 1.0),
    alpha: float = 0.4,
) -> torch.Tensor:
    if total_patches == 0 or masked_indices.numel() == 0:
        return img

    C, H, W = img.shape
    assert C >= 3, "Need at least 3 channels for RGB drawing"

    grid_w = W // pw
    device = img.device

    patch_mask = torch.zeros((1, 1, H, W), device=device)

    for idx in masked_indices.view(-1).tolist():
        if idx < 0 or idx >= total_patches:
            continue
        gy = idx // grid_w
        gx = idx % grid_w
        y0 = gy * ph
        y1 = min((gy + 1) * ph, H)
        x0 = gx * pw
        x1 = min((gx + 1) * pw, W)
        patch_mask[:, :, y0:y1, x0:x1] = 1.0

    mask = patch_mask[0]
    color_tensor = torch.tensor(color, device=device).view(3, 1, 1)

    out = img.clone()
    out = out * (1.0 - alpha * mask) + color_tensor * (alpha * mask)
    return out


@torch.no_grad()
def save_preview(
    step: int,
    out_dir: str,
    views: torch.Tensor,
    recon_views: torch.Tensor,
    loss_mask: torch.Tensor,
    visible_views: torch.Tensor,
    camera_names: Sequence[str],
    patch_h: int,
    patch_w: int,
    frame_rgb: int = 3,
):
    """
    Saves 3-view previews:
      - orig_view_i
      - masked_view_i
      - recon_view_i
    """
    import torchvision.utils as vutils

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    B, V, C, H, W = views.shape
    assert V == len(camera_names), f"camera_names length {len(camera_names)} != views V={V}"

    for vidx, cam_name in enumerate(camera_names):
        safe_name = cam_name.replace("/", "_")

        orig = views[0, vidx, -frame_rgb:].clamp(0, 1)
        rec = recon_views[0, vidx, -frame_rgb:].clamp(0, 1)

        masked_idx = torch.where(loss_mask[0, vidx])[0]
        masked = _draw_patch_boxes_single(
            orig,
            masked_idx,
            ph=patch_h,
            pw=patch_w,
            total_patches=loss_mask.shape[-1],
        )

        suffix = "visible" if bool(visible_views[0, vidx].item()) else "dropped"

        vutils.save_image(orig, os.path.join(out_dir, f"step_{step:06d}_orig_view{vidx}_{safe_name}.png"))
        vutils.save_image(masked, os.path.join(out_dir, f"step_{step:06d}_mask_view{vidx}_{safe_name}_{suffix}.png"))
        vutils.save_image(rec, os.path.join(out_dir, f"step_{step:06d}_recon_view{vidx}_{safe_name}.png"))

        if wandb.run is not None:
            wandb.log(
                {
                    f"preview/{safe_name}/orig": wandb.Image(orig.detach().cpu()),
                    f"preview/{safe_name}/mask": wandb.Image(masked.detach().cpu()),
                    f"preview/{safe_name}/recon": wandb.Image(rec.detach().cpu()),
                    f"preview/{safe_name}/visible": int(bool(visible_views[0, vidx].item())),
                },
                step=step,
            )


# --------------------------- curriculum ---------------------------

PATTERN_TRIPLE = 0
PATTERN_SINGLE = 1


@dataclass
class CurriculumConfig:
    phase_a_end: float
    phase_b_end: float

    phase_a_probs: Tuple[float, float]
    phase_b_probs: Tuple[float, float]
    phase_c_probs: Tuple[float, float]

    triple_mask_range: Tuple[float, float]
    single_mask_range_early: Tuple[float, float]
    single_mask_range_late: Tuple[float, float]

    single_late_start_frac: float
    mask_sampling: str  # uniform | fixed

    pattern_weight_triple: float
    pattern_weight_single: float



def _normalize_probs(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    assert arr.shape == (2,), f"Expected 2 probs (triple,single), got {vals}"
    assert np.all(arr >= 0), f"Probabilities must be non-negative, got {vals}"
    s = float(arr.sum())
    assert s > 0, f"Probability sum must be positive, got {vals}"
    arr = arr / s
    return float(arr[0]), float(arr[1])


def _sample_ratio(
    n: int,
    lo: float,
    hi: float,
    device: torch.device,
    mode: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    lo = float(max(0.0, min(1.0, lo)))
    hi = float(max(0.0, min(1.0, hi)))
    if hi < lo:
        lo, hi = hi, lo

    if mode == "fixed":
        return torch.full((n,), (lo + hi) * 0.5, device=device, dtype=dtype)

    # default uniform
    return torch.empty((n,), device=device, dtype=dtype).uniform_(lo, hi)


def sample_view_pattern(
    *,
    batch_size: int,
    step: int,
    total_steps: int,
    num_views: int,
    cfg: CurriculumConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor | str]:
    """
    For each sample, picks one of: triple / single and returns per-sample:
      - visible_views [B,V] bool
      - mask_ratios  [B,V] float
      - pattern_ids  [B]
      - pattern_weights [B]
    """
    assert num_views == 3, f"This curriculum is defined for 3 views, got {num_views}"

    progress = float(step) / float(max(total_steps, 1))

    if progress <= cfg.phase_a_end:
        probs = cfg.phase_a_probs
        phase_name = "A"
    elif progress <= cfg.phase_b_end:
        probs = cfg.phase_b_probs
        phase_name = "B"
    else:
        probs = cfg.phase_c_probs
        phase_name = "C"

    prob_tensor = torch.tensor(probs, device=device, dtype=torch.float32)
    pattern_choices = torch.tensor([PATTERN_TRIPLE, PATTERN_SINGLE], device=device, dtype=torch.long)
    sampled_pattern_idx = torch.multinomial(prob_tensor, num_samples=batch_size, replacement=True).long()
    pattern_ids = pattern_choices[sampled_pattern_idx]

    visible_views = torch.zeros((batch_size, num_views), device=device, dtype=torch.bool)
    mask_ratios = torch.ones((batch_size, num_views), device=device, dtype=dtype)

    single_range = cfg.single_mask_range_late if progress >= cfg.single_late_start_frac else cfg.single_mask_range_early

    for b in range(batch_size):
        pid = int(pattern_ids[b].item())

        if pid == PATTERN_TRIPLE:
            visible_views[b] = True
            mask_ratios[b] = _sample_ratio(
                num_views,
                cfg.triple_mask_range[0],
                cfg.triple_mask_range[1],
                device,
                cfg.mask_sampling,
                dtype,
            )

        elif pid == PATTERN_SINGLE:
            keep_idx = int(torch.randint(low=0, high=num_views, size=(1,), device=device).item())
            visible_views[b, keep_idx] = True
            ratio = _sample_ratio(
                1,
                single_range[0],
                single_range[1],
                device,
                cfg.mask_sampling,
                dtype,
            )[0]
            mask_ratios[b, keep_idx] = ratio
            for vidx in range(num_views):
                if vidx != keep_idx:
                    mask_ratios[b, vidx] = 1.0

        else:
            raise AssertionError(f"Unknown pattern id {pid}")

    assert bool((visible_views.sum(dim=1) >= 1).all()), "Every sample must keep at least one visible view"

    pattern_weights = torch.full((batch_size,), float(cfg.pattern_weight_triple), device=device, dtype=dtype)
    pattern_weights[pattern_ids == PATTERN_SINGLE] = float(cfg.pattern_weight_single)

    return {
        "phase": phase_name,
        "pattern_ids": pattern_ids,
        "visible_views": visible_views,
        "mask_ratios": mask_ratios,
        "pattern_weights": pattern_weights,
    }


# --------------------------- main ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env_name", type=str, default="button-press-topdown-v3")

    # 3-camera setup.
    p.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["gripperPOV", "corner", "corner2"],
        help="Three camera names or ids, e.g. --camera_names gripperPOV corner corner2",
    )

    # Model
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--decoder_dim", type=int, default=256)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--decoder_heads", type=int, default=4)

    # Train
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--decoder_lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1_000)
    p.add_argument("--vis_every", type=int, default=10_000)
    p.add_argument("--out", type=str, default="vtmae_pretrained_3cam.pt")
    p.add_argument("--preview_dir", type=str, default=None)

    # Schedule / curriculum
    p.add_argument("--schedule", type=str, default="default_3cam", choices=["default_3cam"])
    p.add_argument("--phase_a_end", type=float, default=0.20)
    p.add_argument("--phase_b_end", type=float, default=0.60)

    p.add_argument(
        "--phase_a_pattern_probs",
        type=float,
        nargs=2,
        default=[0.70, 0.05],
        help="Triple/single sampling weights for phase A.",
    )
    p.add_argument(
        "--phase_b_pattern_probs",
        type=float,
        nargs=2,
        default=[0.40, 0.20],
        help="Triple/single sampling weights for phase B.",
    )
    p.add_argument(
        "--phase_c_pattern_probs",
        type=float,
        nargs=2,
        default=[0.20, 0.45],
        help="Triple/single sampling weights for phase C.",
    )

    p.add_argument("--triple_mask_range", type=float, nargs=2, default=[0.75, 0.75])
    p.add_argument("--single_mask_range_early", type=float, nargs=2, default=[0.75, 0.75])
    p.add_argument("--single_mask_range_late", type=float, nargs=2, default=[0.75, 0.75])
    p.add_argument("--single_late_start_frac", type=float, default=0.60)
    p.add_argument("--mask_sampling", type=str, default="uniform", choices=["uniform", "fixed"])

    # Loss weighting
    p.add_argument("--pattern_weight_triple", type=float, default=1.0)
    p.add_argument("--pattern_weight_single", type=float, default=1.50)
    p.add_argument("--cross_view_loss_weight", type=float, default=1.5)

    # Optional latent alignment
    p.add_argument("--use_alignment_loss", dest="use_alignment_loss", action="store_true", default=True)
    p.add_argument("--no_alignment_loss", dest="use_alignment_loss", action="store_false")
    p.add_argument("--alignment_weight", type=float, default=0.1)
    p.add_argument("--alignment_mode", type=str, default="cosine", choices=["cosine", "mse"])

    # WandB
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vtmae-3cam")
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

    assert 0.0 < args.phase_a_end < args.phase_b_end < 1.0, (
        f"Need 0 < phase_a_end < phase_b_end < 1, got {args.phase_a_end}, {args.phase_b_end}"
    )

    cfg = CurriculumConfig(
        phase_a_end=args.phase_a_end,
        phase_b_end=args.phase_b_end,
        phase_a_probs=_normalize_probs(args.phase_a_pattern_probs),
        phase_b_probs=_normalize_probs(args.phase_b_pattern_probs),
        phase_c_probs=_normalize_probs(args.phase_c_pattern_probs),
        triple_mask_range=(float(args.triple_mask_range[0]), float(args.triple_mask_range[1])),
        single_mask_range_early=(float(args.single_mask_range_early[0]), float(args.single_mask_range_early[1])),
        single_mask_range_late=(float(args.single_mask_range_late[0]), float(args.single_mask_range_late[1])),
        single_late_start_frac=float(args.single_late_start_frac),
        mask_sampling=str(args.mask_sampling),
        pattern_weight_triple=float(args.pattern_weight_triple),
        pattern_weight_single=float(args.pattern_weight_single),
    )

    out_images_dir = args.preview_dir or f"{args.schedule}_{'_'.join(camera_names)}_mw"
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

    print(f"[VTMAE-3CAM] channels={channels}, views={len(camera_names)}, size=({H},{W})")
    print(f"[VTMAE-3CAM] cameras={camera_names}")

    v = MultiViewVST(
        image_size=(H, W),
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        channels=channels,
        num_views=len(camera_names),
        frame_stack=args.frame_stack,
    )

    mae = MultiViewVSMAE(
        encoder=v,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        use_sincosmod_encodings=True,
        frame_stack=args.frame_stack,
        num_views=len(camera_names),
    ).to(device)

    print(f"[Params] MultiViewVST total: {fmt(count_params(v))} | trainable: {fmt(count_params(v, True))}")
    print(f"[Params] MultiViewVSMAE total: {fmt(count_params(mae))} | trainable: {fmt(count_params(mae, True))}")
    print(
        "[Params breakdown] "
        f"encoder={fmt(count_params(mae.encoder))} "
        f"decoder={fmt(count_params(mae.decoder))} "
        f"heads={fmt(count_params(mae.to_view_pixels))}"
    )

    encoder_params = list(mae.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in mae.parameters() if id(p) not in encoder_param_ids]

    encoder_opt = torch.optim.AdamW(encoder_params, lr=args.encoder_lr, weight_decay=0.05)
    decoder_opt = torch.optim.AdamW(decoder_params, lr=args.decoder_lr, weight_decay=0.0)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    mae.train()
    running = None
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        views = collect_batch(env, args.batch_size, view_keys, device)

        pattern = sample_view_pattern(
            batch_size=args.batch_size,
            step=step,
            total_steps=args.steps,
            num_views=len(camera_names),
            cfg=cfg,
            device=device,
            dtype=views.dtype,
        )

        need_debug = bool(args.vis_every and (step % args.vis_every == 0))

        out = mae(
            views,
            visible_views=pattern["visible_views"],
            mask_ratios=pattern["mask_ratios"],
            pattern_ids=pattern["pattern_ids"],
            pattern_weights=pattern["pattern_weights"],
            cross_view_loss_weight=args.cross_view_loss_weight,
            return_breakdown=True,
            return_debug=need_debug,
        )

        recon_loss = out["total"]
        alignment_loss = torch.tensor(0.0, device=device)

        if args.use_alignment_loss:
            # Anchor subset latents to the full 3-view latent for the same scene batch.
            z_subset = mae.get_embeddings(views, visible_views=pattern["visible_views"], eval=False)
            z_full = mae.get_embeddings(
                views,
                visible_views=torch.ones_like(pattern["visible_views"]),
                eval=False,
            ).detach()

            if args.alignment_mode == "cosine":
                alignment_loss = 1.0 - F.cosine_similarity(z_subset, z_full, dim=-1).mean()
            else:
                alignment_loss = F.mse_loss(z_subset, z_full)

        total_loss = recon_loss + float(args.alignment_weight) * alignment_loss

        encoder_opt.zero_grad(set_to_none=True)
        decoder_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        encoder_opt.step()
        decoder_opt.step()

        val_total = float(total_loss.detach().cpu().item())
        val_recon = float(recon_loss.detach().cpu().item())
        val_align = float(alignment_loss.detach().cpu().item())
        running = val_total if running is None else (0.99 * running + 0.01 * val_total)

        if step % args.log_every == 0:
            pattern_ids = pattern["pattern_ids"]
            counts = torch.bincount(pattern_ids, minlength=2)

            view_mse = out["per_view_mse"].detach().cpu().tolist()
            vis_loss = float(out["visible_loss"].detach().cpu().item())
            cross_loss = float(out["cross_view_loss"].detach().cpu().item())

            mask_means = []
            for vidx in range(len(camera_names)):
                vis = pattern["visible_views"][:, vidx]
                if bool(vis.any()):
                    mask_means.append(float(pattern["mask_ratios"][vis, vidx].mean().detach().cpu().item()))
                else:
                    mask_means.append(1.0)

            print(
                f"[{step}/{args.steps}] phase={pattern['phase']} "
                f"total={val_total:.4f} ema={running:.4f} recon={val_recon:.4f} align={val_align:.4f} "
                f"v0={view_mse[0]:.4f} v1={view_mse[1]:.4f} v2={view_mse[2]:.4f} "
                f"visible={vis_loss:.4f} cross={cross_loss:.4f} "
                f"patterns(t/s)={int(counts[0])}/{int(counts[1])} "
                f"mask(v0/v1/v2)={mask_means[0]:.2f}/{mask_means[1]:.2f}/{mask_means[2]:.2f}",
                flush=True,
            )

            if args.wandb:
                wb = {
                    "loss/total": val_total,
                    "loss/recon": val_recon,
                    "loss/alignment": val_align,
                    "loss/visible": vis_loss,
                    "loss/cross_view": cross_loss,
                    "loss/view0": float(out["view0_mse"].detach().cpu().item()),
                    "loss/view1": float(out["view1_mse"].detach().cpu().item()),
                    "loss/view2": float(out["view2_mse"].detach().cpu().item()),
                    "loss/pattern_weight_mean": float(out["pattern_weight_mean"].detach().cpu().item()),
                    "stats/phase": {"A": 0, "B": 1, "C": 2}[str(pattern["phase"])],
                    "stats/pattern_triple_count": int(counts[0].item()),
                    "stats/pattern_single_count": int(counts[1].item()),
                    "stats/mask_ratio_view0": mask_means[0],
                    "stats/mask_ratio_view1": mask_means[1],
                    "stats/mask_ratio_view2": mask_means[2],
                }

                if "pattern_loss_triple" in out:
                    wb["loss/pattern_triple"] = float(out["pattern_loss_triple"].detach().cpu().item())
                    wb["loss/pattern_single"] = float(out["pattern_loss_single"].detach().cpu().item())

                wandb.log(wb, step=step)

        if args.save_every and (step % args.save_every == 0):
            torch.save(mae.state_dict(), args.out)

        if need_debug and "debug" in out:
            debug = out["debug"]
            recon_views = composite_recon_from_debug(mae, views, debug)
            save_preview(
                step=step,
                out_dir=out_images_dir,
                views=views,
                recon_views=recon_views,
                loss_mask=debug["loss_mask"],
                visible_views=debug["visible_views"],
                camera_names=camera_names,
                patch_h=mae.patch_height,
                patch_w=mae.patch_width,
                frame_rgb=3,
            )

    torch.save(mae.state_dict(), args.out)
    if args.wandb:
        wandb.save(args.out)
    print(f"Done. Saved weights to {args.out}")


if __name__ == "__main__":
    main()


"""
Example 3-camera run:

python pretrain_vtmae.py \
  --camera_names gripperPOV corner corner2 \
  --frame_stack 3 \
  --action_repeat 2 \
  --patch_size 6 \
  --steps 100000 \
  --batch_size 32 \
  --schedule default_3cam \
  --wandb \
  --wandb_project vtmae-3cam \
  --wandb_run singlecam_to_allcams
"""
