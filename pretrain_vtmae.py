# =========================
# extra/pretrain_vtmae.py
# (NO segmentation anywhere)
# =========================
from __future__ import annotations
import os
import argparse
from typing import Dict, Tuple

# ---- Headless MuJoCo (must be before dm_control import) ----
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.pop("DISPLAY", None)

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from pathlib import Path
from einops import rearrange

from metaworld_dm_env import make_metaworld
from pretrain_models import VST, VSMAE


# --------------------------- utils ---------------------------

def set_seed(seed: int):
    import random
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def _get_obs_dict(ts) -> Dict[str, np.ndarray]:
    """
    Expect env to provide:
      - obs["pixels"]      : primary RGB
      - obs["pixels_aux"]  : auxiliary RGB from a second camera
    """
    obs = ts.observation if hasattr(ts, "observation") else ts
    assert "pixels" in obs and "pixels_aux" in obs, f"obs keys={list(obs.keys())}"
    return {"pixels": obs["pixels"], "pixels_aux": obs["pixels_aux"]}

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
        if obs_item.shape[-1] in (1, 3, 4):  # assume HWC
            return torch.from_numpy(obs_item).permute(2, 0, 1).contiguous()
        return torch.from_numpy(obs_item)  # assume CHW
    raise AssertionError(f"unexpected obs shape {obs_item.shape}")

@torch.no_grad()
def _sample_action(env) -> np.ndarray:
    spec = env.action_spec()
    shape = spec.shape if getattr(spec, "shape", ()) else (1,)
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)

def peek_channels(env) -> Tuple[int, int, int, int]:
    ts = env.reset()
    obs = _get_obs_dict(ts)
    img_chw = _to_chw_stacked(obs["pixels"])
    aux_chw = _to_chw_stacked(obs["pixels_aux"])
    Ci, H, W = img_chw.shape
    Ca, H2, W2 = aux_chw.shape
    assert (H, W) == (H2, W2), "pixels/pixels_aux spatial size mismatch"
    return Ci, Ca, H, W

def collect_batch(env, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, auxs = [], []
    ts = env.reset()
    for _ in range(batch_size):
        obs = _get_obs_dict(ts)
        imgs.append(_to_chw_stacked(obs["pixels"]))        # [Ci,H,W]
        auxs.append(_to_chw_stacked(obs["pixels_aux"]))    # [Ca,H,W]
        ts = env.step(_sample_action(env))
        if hasattr(ts, "last") and ts.last():
            ts = env.reset()
    img = torch.stack(imgs, dim=0).float().to(device) / 255.0
    aux = torch.stack(auxs, dim=0).float().to(device) / 255.0
    return img, aux


# ---------- reconstruction (no masking) ----------
@torch.no_grad()
def reconstruct_full(mae: VSMAE, img: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(mae.parameters()).device
    enc = mae.encoder
    B, Ci, H, W = img.shape
    _, Ca, _, _ = aux.shape

    Hp = enc.image_height // enc.image_patch_height
    Wp = enc.image_width  // enc.image_patch_width
    Ha = enc.aux_height // enc.aux_patch_height
    Wa = enc.aux_width  // enc.aux_patch_width

    # ---- image tokens ----
    if Ci > 0:
        feat = mae.early_conv_vision(img)
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, (Hp, Wp))
            feat = feat.flatten(2).transpose(1, 2)  # [B,Hp*Wp,D]
        ti = feat
        if mae.use_sincosmod_encodings:
            ti = ti + mae.encoder_modality_embedding(torch.tensor(0, device=device)) + mae.image_enc_pos_embedding
    else:
        ti = torch.zeros((B, 0, mae.encoder_dim), device=device)

    # ---- aux tokens ----
    if Ca > 0:
        feat = mae.early_conv_aux(aux)
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, (Ha, Wa))
            feat = feat.flatten(2).transpose(1, 2)  # [B,Ha*Wa,D]
        ta = feat
        if mae.use_sincosmod_encodings:
            ta = ta + mae.encoder_modality_embedding(torch.tensor(1, device=device)) + mae.aux_enc_pos_embedding
    else:
        ta = torch.zeros((B, 0, mae.encoder_dim), device=device)

    tokens = torch.cat([ti, ta], dim=1)
    Ni = ti.shape[1]
    Na = ta.shape[1]

    enc_out = enc.transformer(tokens)
    dec_in = mae.enc_to_dec(enc_out)

    if mae.use_sincosmod_encodings:
        if Ni > 0:
            dec_in[:, :Ni] += mae.decoder_modality_embedding(torch.tensor(0, device=device)) + mae.image_dec_pos_embedding
        if Na > 0:
            dec_in[:, Ni:] += mae.decoder_modality_embedding(torch.tensor(1, device=device)) + mae.aux_dec_pos_embedding

    dec_out = mae.decoder(dec_in)
    pred_img_patches = mae.to_pixels(dec_out[:, :Ni]) if Ni > 0 else None
    pred_aux_patches = mae.to_aux_pixels(dec_out[:, Ni:]) if Na > 0 else None

    def unpatchify(patches, C, H, W, ph, pw):
        h = H // ph; w = W // pw
        return rearrange(patches, "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                         h=h, w=w, ph=ph, pw=pw, c=C)

    rec_img = torch.zeros_like(img)
    if pred_img_patches is not None:
        rec_img = unpatchify(
            pred_img_patches, enc.image_channels, enc.image_height, enc.image_width,
            enc.image_patch_height, enc.image_patch_width
        ).clamp(0, 1)

    rec_aux = torch.zeros_like(aux)
    if pred_aux_patches is not None:
        rec_aux = unpatchify(
            pred_aux_patches, enc.aux_channels, enc.aux_height, enc.aux_width,
            enc.aux_patch_height, enc.aux_patch_width
        ).clamp(0, 1)

    return rec_img, rec_aux

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

    grid_h = H // ph
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

    mask = patch_mask[0]  # [1,H,W]
    color_tensor = torch.tensor(color, device=device).view(3, 1, 1)

    out = img.clone()
    out = out * (1.0 - alpha * mask) + color_tensor * (alpha * mask)
    return out

@torch.no_grad()
def save_preview(
    step: int,
    out_dir: str,
    img_aug: torch.Tensor,
    aux_aug: torch.Tensor,
    rec_img: torch.Tensor,
    rec_aux: torch.Tensor,
    m_idx_img: torch.Tensor,
    m_idx_aux: torch.Tensor,
    Ni: int,
    Na: int,
    patch_h: int,
    patch_w: int,
    frame_rgb=3,
    frame_aux=3,
):
    import torchvision.utils as vutils
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    b0_img = img_aug[0, :frame_rgb].clamp(0, 1)
    b0_aux = aux_aug[0, :frame_aux].clamp(0, 1)
    r0_img = rec_img[0, :frame_rgb].clamp(0, 1)
    r0_aux = rec_aux[0, :frame_aux].clamp(0, 1)

    boxed_rgb = _draw_patch_boxes_single(b0_img, m_idx_img[0], patch_h, patch_w, Ni) if (Ni > 0 and m_idx_img.numel() > 0) else b0_img
    boxed_aux = _draw_patch_boxes_single(b0_aux, m_idx_aux[0], patch_h, patch_w, Na) if (Na > 0 and m_idx_aux.numel() > 0) else b0_aux

    vutils.save_image(b0_img,    os.path.join(out_dir, f"step_{step:06d}_aug_rgb.png"))
    vutils.save_image(b0_aux,    os.path.join(out_dir, f"step_{step:06d}_aug_aux.png"))
    vutils.save_image(boxed_rgb, os.path.join(out_dir, f"step_{step:06d}_masked_rgb.png"))
    vutils.save_image(boxed_aux, os.path.join(out_dir, f"step_{step:06d}_masked_aux.png"))
    vutils.save_image(r0_img,    os.path.join(out_dir, f"step_{step:06d}_rec_rgb.png"))
    vutils.save_image(r0_aux,    os.path.join(out_dir, f"step_{step:06d}_rec_aux.png"))

    if wandb.run is not None:
        wandb.log(
            {
                "preview/aug_rgb": wandb.Image(b0_img.detach().cpu()),
                "preview/aug_aux": wandb.Image(b0_aux.detach().cpu()),
                "preview/masked_rgb": wandb.Image(boxed_rgb.detach().cpu()),
                "preview/masked_aux": wandb.Image(boxed_aux.detach().cpu()),
                "preview/rec_rgb": wandb.Image(r0_img.detach().cpu()),
                "preview/rec_aux": wandb.Image(r0_aux.detach().cpu()),
            },
            step=step,
        )


# --------------------------- main ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)

    # Camera names OR camera indices (0..5). If you pass an int, we map it to the name below.
    p.add_argument("--camera_main", type=str, default="gripperPOV")
    p.add_argument("--camera_aux", type=str, default="corner")

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
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--vis_every", type=int, default=10_000)
    p.add_argument("--out", type=str, default="vtmae_pretrained.pt")

    # WandB
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vtmae-only")
    p.add_argument("--wandb_run", type=str, default=None)

    # masking / aux loss
    p.add_argument("--masking_ratio_a", type=float, default=0.75)
    p.add_argument("--masking_ratio_b", type=float, default=0.75)
    p.add_argument("--aux_loss", type=float, default=1)

    args = p.parse_args()

    camera_map = {
        "0": "topview",
        "1": "corner",
        "2": "corner2",
        "3": "corner3",
        "4": "behindGripper",
        "5": "gripperPOV",
    }
    cam_main = camera_map.get(str(args.camera_main), args.camera_main)
    cam_aux  = camera_map.get(str(args.camera_aux),  args.camera_aux)

    out_images_dir = f"{args.masking_ratio_a}_{args.masking_ratio_b}_{args.aux_loss}_mw"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_metaworld(
        name="button-press-topdown-v3",
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        camera_name=cam_main,
        camera_aux_name=cam_aux,
        add_aux_pixels_to_obs=True,
    )

    Ci, Ca, H, W = peek_channels(env)
    assert H % args.patch_size == 0 and W % args.patch_size == 0, \
        f"({H},{W}) must be divisible by patch_size={args.patch_size}"
    print(f"[VTMAE] image_channels={Ci}, aux_channels={Ca}, size=({H},{W})")
    print(f"[VTMAE] cameras: main={cam_main} aux={cam_aux}")

    v = VST(
        image_size=(H, W),
        aux_size=(H, W),
        image_patch_size=args.patch_size,
        aux_patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        image_channels=Ci,
        aux_channels=Ca,
        frame_stack=args.frame_stack,
    )

    mae = VSMAE(
        encoder=v,
        decoder_dim=args.decoder_dim,
        masking_ratio_a=args.masking_ratio_a,
        masking_ratio_b=args.masking_ratio_b,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        use_sincosmod_encodings=True,
        frame_stack=args.frame_stack,
        auxloss_multiplier=args.aux_loss,
    ).to(device)

    enc_params = list(mae.encoder.parameters())
    enc_params += list(mae.early_conv_vision.parameters())
    enc_params += list(mae.early_conv_aux.parameters())
    encoder_opt = torch.optim.AdamW(enc_params, lr=args.encoder_lr, weight_decay=0.05)

    dec_params = (
        list(mae.decoder.parameters())
        + list(mae.to_pixels.parameters())
        + list(mae.to_aux_pixels.parameters())
        + list(mae.decoder_modality_embedding.parameters())
    )
    decoder_opt = torch.optim.AdamW(dec_params, lr=args.decoder_lr, weight_decay=0.0)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    mae.train()
    running = None
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        img, aux = collect_batch(env, args.batch_size, device)
        img_aug = img
        aux_aug = aux

        out = mae({"image": img_aug, "image_aux": aux_aug}, return_breakdown=True)
        recon_loss = out["total"]

        rgb_mse = out["rgb_mse"].detach().item()
        aux_mse = out["aux_mse"].detach().item()
        aux_weight = out["aux_weight"]

        encoder_opt.zero_grad(set_to_none=True)
        decoder_opt.zero_grad(set_to_none=True)
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        encoder_opt.step()
        decoder_opt.step()

        val = float(recon_loss.detach().cpu().item())
        running = val if running is None else (0.99 * running + 0.01 * val)

        if step % args.log_every == 0:
            print(
                f"[{step}/{args.steps}] "
                f"total={val:.4f} ema={running:.4f} "
                f"rgb_mse={rgb_mse:.4f} aux_mse={aux_mse:.4f} "
                f"aux_weight={aux_weight:.3f}",
                flush=True,
            )
            if args.wandb:
                wandb.log(
                    {
                        "mae_total": val,
                        "mae_total_ema": running,
                        "rgb_mse": rgb_mse,
                        "aux_mse": aux_mse,
                        "aux_weight": aux_weight,
                        "aux_contrib_frac": (aux_weight * aux_mse) / (val + 1e-8),
                    },
                    step=step,
                )

        if args.save_every and (step % args.save_every == 0):
            torch.save(mae.state_dict(), args.out)

        if step % args.vis_every == 0:
            mae.eval()
            with torch.no_grad():
                _, dbg = mae({"image": img_aug, "image_aux": aux_aug}, return_debug=True)
                m_idx_img = dbg["m_idx_img"]
                m_idx_aux = dbg["m_idx_aux"]
                Ni = dbg["Ni"]
                Na = dbg["Na"]
                rec_img, rec_aux = reconstruct_full(mae, img_aug, aux_aug)
            mae.train()

            save_preview(
                step, out_images_dir,
                img_aug, aux_aug, rec_img, rec_aux,
                m_idx_img=m_idx_img, m_idx_aux=m_idx_aux,
                Ni=Ni, Na=Na,
                patch_h=mae.encoder.image_patch_height,
                patch_w=mae.encoder.image_patch_width,
                frame_rgb=3, frame_aux=3,
            )

    torch.save(mae.state_dict(), args.out)
    if args.wandb:
        wandb.save(args.out)
    print(f"Done. Saved weights to {args.out}")

if __name__ == "__main__":
    main()


"""

python pretrain_vtmae.py --env_type mw --seg_loss 0.01 --wandb --wandb_run mw_seg0p01



commands = [

    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 0.01 --wandb --wandb_run mw_seg0p01",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 0.03 --wandb --wandb_run mw_seg0p03",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 0.10 --wandb --wandb_run mw_seg0p10",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 0.30 --wandb --wandb_run mw_seg0p30",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 0.50 --wandb --wandb_run mw_seg0p50",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 0.75 --wandb --wandb_run mw_seg0p75",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 1.00 --wandb --wandb_run mw_seg1p00",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 1.50 --wandb --wandb_run mw_seg1p50",
    "python extra/pretrain_vtmae.py --env_type dmc --seg_loss 2.00 --wandb --wandb_run mw_seg2p00",

    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 0.01 --wandb --wandb_run mw_seg0p01",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 0.03 --wandb --wandb_run mw_seg0p03",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 0.10 --wandb --wandb_run mw_seg0p10",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 0.30 --wandb --wandb_run mw_seg0p30",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 0.50 --wandb --wandb_run mw_seg0p50",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 0.75 --wandb --wandb_run mw_seg0p75",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 1.00 --wandb --wandb_run mw_seg1p00",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 1.50 --wandb --wandb_run mw_seg1p50",
    "python extra/pretrain_vtmae.py --env_type mw --seg_loss 2.00 --wandb --wandb_run mw_seg2p00",

]




0: topview
1: corner
2: corner2
3: corner3
4: behindGripper
5: gripperPOV


python pretrain_vtmae.py \
  --camera_main topview \
  --camera_aux corner \
  --wandb \
  --wandb_project vtmae-only \
  --wandb_run mw_rgb_rgb_gripper_corner


"""