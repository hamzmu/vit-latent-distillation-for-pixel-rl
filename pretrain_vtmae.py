# =========================
# extra/pretrain_vtmae.py
# (NO segmentation anywhere)
# ViT-only (NO CNN anywhere)
# Preview saves ONLY 6 images per vis step:
#   - orig modality1 + modality2
#   - masked modality1 + modality2 (overlay)
#   - reconstructed modality1 + modality2 (COMPOSITE recon)
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


# --------------------------- patch helpers ---------------------------

def patchify(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """
    x: [B,C,H,W] -> [B, N, ph*pw*C]
    """
    B, C, H, W = x.shape
    assert H % ph == 0 and W % pw == 0
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)

def unpatchify(patches: torch.Tensor, C: int, H: int, W: int, ph: int, pw: int) -> torch.Tensor:
    """
    patches: [B, N, ph*pw*C] -> [B,C,H,W]
    """
    h = H // ph
    w = W // pw
    return rearrange(patches, "b (h w) (ph pw c) -> b c (h ph) (w pw)", h=h, w=w, ph=ph, pw=pw, c=C)


# --------------------------- reconstructions ---------------------------

@torch.no_grad()
def composite_recon_from_mask(
    mae: VSMAE,
    img: torch.Tensor,
    aux: torch.Tensor,
    m_idx_img: torch.Tensor,
    m_idx_aux: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MAE-style COMPOSITE reconstruction:
      - visible patches copied from input
      - masked patches replaced with model predictions
    Uses the SAME masked indices you trained on (from return_debug).
    (ViT-only: patch embedding tokens, encode only unmasked, decode full)
    """
    device = next(mae.parameters()).device
    enc = mae.encoder

    B, Ci, H, W = img.shape
    _, Ca, _, _ = aux.shape

    ph_i, pw_i = enc.image_patch_height, enc.image_patch_width
    ph_a, pw_a = enc.aux_patch_height, enc.aux_patch_width

    # patchify inputs (targets + composite base)
    img_patches = patchify(img, ph_i, pw_i) if Ci > 0 else torch.zeros((B, 0, 0), device=device)
    aux_patches = patchify(aux, ph_a, pw_a) if Ca > 0 else torch.zeros((B, 0, 0), device=device)

    Ni = img_patches.shape[1]
    Na = aux_patches.shape[1]
    Nt = Ni + Na

    # tokens (same as training)
    if Ci > 0:
        ti = mae.image_patch_embed(img)  # [B,Ni,D]
        if mae.use_sincosmod_encodings:
            ti = ti + mae.encoder_modality_embedding.weight[0] + mae.image_enc_pos_embedding
    else:
        ti = torch.zeros((B, 0, mae.encoder_dim), device=device)

    if Ca > 0:
        ta = mae.aux_patch_embed(aux)    # [B,Na,D]
        if mae.use_sincosmod_encodings:
            ta = ta + mae.encoder_modality_embedding.weight[1] + mae.aux_enc_pos_embedding
    else:
        ta = torch.zeros((B, 0, mae.encoder_dim), device=device)

    tokens = torch.cat([ti, ta], dim=1)  # [B,Nt,D]

    batch_range = torch.arange(B, device=device)[:, None]

    m_img = m_idx_img if Ni > 0 else torch.zeros((B, 0), dtype=torch.long, device=device)
    m_aux = m_idx_aux if Na > 0 else torch.zeros((B, 0), dtype=torch.long, device=device)

    m_idx = torch.cat([m_img, m_aux + Ni], dim=1)  # global masked positions [B, M]

    # unmasked indices = complement
    if Nt > 0:
        keep = torch.ones((B, Nt), dtype=torch.bool, device=device)
        if m_idx.numel() > 0:
            keep[batch_range, m_idx] = False
        u_idx = keep.nonzero(as_tuple=False).view(B, -1, 2)[:, :, 1]  # [B, Nu]
    else:
        u_idx = torch.zeros((B, 0), dtype=torch.long, device=device)

    # encode only unmasked tokens (same as training)
    enc_in = tokens[batch_range, u_idx]          # [B,Nu,D]
    enc_out = mae.encoder.transformer(enc_in)    # [B,Nu,D]
    dec_tok = mae.enc_to_dec(enc_out)            # [B,Nu,Dd]

    # assemble decoder input with mask tokens
    dec_tokens = torch.zeros(B, Nt, mae.decoder_dim, device=device)
    if u_idx.numel() > 0:
        dec_tokens[batch_range, u_idx] = dec_tok
    if m_idx.numel() > 0:
        dec_tokens[batch_range, m_idx] = mae.mask_token

    if mae.use_sincosmod_encodings:
        if Ni > 0:
            dec_tokens[:, :Ni] += mae.decoder_modality_embedding.weight[0]
            dec_tokens[:, :Ni] += mae.image_dec_pos_embedding
        if Na > 0:
            dec_tokens[:, Ni:] += mae.decoder_modality_embedding.weight[1]
            dec_tokens[:, Ni:] += mae.aux_dec_pos_embedding

    dec_out = mae.decoder(dec_tokens)  # [B,Nt,Dd]

    # predict ONLY masked patches and write into composite
    if Ni > 0 and m_img.numel() > 0:
        pred_img = mae.to_pixels(dec_out[batch_range, m_img])  # [B,Mi,patchdim]
        img_patches = img_patches.clone()
        img_patches[batch_range, m_img] = pred_img

    if Na > 0 and m_aux.numel() > 0:
        pred_aux = mae.to_aux_pixels(dec_out[batch_range, m_aux + Ni])
        aux_patches = aux_patches.clone()
        aux_patches[batch_range, m_aux] = pred_aux

    # unpatchify composite
    comp_img = torch.zeros_like(img)
    if Ni > 0:
        comp_img = unpatchify(img_patches, enc.image_channels, enc.image_height, enc.image_width, ph_i, pw_i).clamp(0, 1)

    comp_aux = torch.zeros_like(aux)
    if Na > 0:
        comp_aux = unpatchify(aux_patches, enc.aux_channels, enc.aux_height, enc.aux_width, ph_a, pw_a).clamp(0, 1)

    return comp_img, comp_aux


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

    mask = patch_mask[0]  # [1,H,W]
    color_tensor = torch.tensor(color, device=device).view(3, 1, 1)

    out = img.clone()
    out = out * (1.0 - alpha * mask) + color_tensor * (alpha * mask)
    return out


@torch.no_grad()
def save_preview(
    step: int,
    out_dir: str,
    img: torch.Tensor,
    aux: torch.Tensor,
    recon_img: torch.Tensor,
    recon_aux: torch.Tensor,
    m_idx_img: torch.Tensor,
    m_idx_aux: torch.Tensor,
    Ni: int,
    Na: int,
    patch_h: int,
    patch_w: int,
    frame_rgb=3,
    frame_aux=3,
):
    """
    Saves ONLY 6 images:
      - orig modality1 + modality2
      - masked modality1 + modality2 (box overlay)
      - reconstructed modality1 + modality2 (composite recon)
    """
    import torchvision.utils as vutils
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # use LATEST frame in the stack
    orig1 = img[0, -frame_rgb:].clamp(0, 1)
    orig2 = aux[0, -frame_aux:].clamp(0, 1)

    rec1 = recon_img[0, -frame_rgb:].clamp(0, 1)
    rec2 = recon_aux[0, -frame_aux:].clamp(0, 1)

    masked1 = _draw_patch_boxes_single(orig1, m_idx_img[0], patch_h, patch_w, Ni) if (Ni > 0 and m_idx_img.numel() > 0) else orig1
    masked2 = _draw_patch_boxes_single(orig2, m_idx_aux[0], patch_h, patch_w, Na) if (Na > 0 and m_idx_aux.numel() > 0) else orig2

    vutils.save_image(orig1,   os.path.join(out_dir, f"step_{step:06d}_orig_mod1.png"))
    vutils.save_image(orig2,   os.path.join(out_dir, f"step_{step:06d}_orig_mod2.png"))
    vutils.save_image(masked1, os.path.join(out_dir, f"step_{step:06d}_mask_mod1.png"))
    vutils.save_image(masked2, os.path.join(out_dir, f"step_{step:06d}_mask_mod2.png"))
    vutils.save_image(rec1,    os.path.join(out_dir, f"step_{step:06d}_recon_mod1.png"))
    vutils.save_image(rec2,    os.path.join(out_dir, f"step_{step:06d}_recon_mod2.png"))

    if wandb.run is not None:
        wandb.log(
            {
                "preview/orig_mod1": wandb.Image(orig1.detach().cpu()),
                "preview/orig_mod2": wandb.Image(orig2.detach().cpu()),
                "preview/mask_mod1": wandb.Image(masked1.detach().cpu()),
                "preview/mask_mod2": wandb.Image(masked2.detach().cpu()),
                "preview/recon_mod1": wandb.Image(rec1.detach().cpu()),
                "preview/recon_mod2": wandb.Image(rec2.detach().cpu()),
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
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--decoder_lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1_000)
    p.add_argument("--vis_every", type=int, default=1_000)
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

    # ViT-only: encoder optimizer just uses mae.encoder
    encoder_opt = torch.optim.AdamW(list(mae.encoder.parameters()), lr=args.encoder_lr, weight_decay=0.05)

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

        if args.vis_every and (step % args.vis_every == 0):
            mae.eval()
            with torch.no_grad():
                _, dbg = mae({"image": img_aug, "image_aux": aux_aug}, return_debug=True)
                m_idx_img = dbg["m_idx_img"]  # [B, Mi]
                m_idx_aux = dbg["m_idx_aux"]  # [B, Ma]
                Ni = dbg["Ni"]
                Na = dbg["Na"]

                recon_img, recon_aux = composite_recon_from_mask(mae, img_aug, aux_aug, m_idx_img, m_idx_aux)

            mae.train()

            save_preview(
                step, out_images_dir,
                img_aug, aux_aug,
                recon_img, recon_aux,
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






0: topview
1: corner
2: corner2
3: corner3
4: behindGripper
5: gripperPOV

python pretrain_vtmae.py \
  --camera_main topview \
  --camera_aux corner \
  --frame_stack 3 \
  --action_repeat 2 \
  --patch_size 6 \
  --masking_ratio_a 1.0 \
  --masking_ratio_b 0.75 \
  --aux_loss 1.0 \
  --batch_size 32 \
  --wandb \
  --wandb_project vtmae-only \
  --wandb_run mw_vitonly_topview_corner


"""