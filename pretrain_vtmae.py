
from __future__ import annotations
import os
import argparse
from typing import Dict, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.pop("DISPLAY", None)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import wandb
from pathlib import Path
from einops import rearrange

# from sear.environments.dmc import make as make_env
from metaworld_dm_env import make_metaworld
from sear.models.pretrain_models import VST, VSMAE

# --------------------------- utils ---------------------------

def set_seed(seed: int):
    import random
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def _get_obs_dict(ts) -> Dict[str, np.ndarray]:
    obs = ts.observation if hasattr(ts, "observation") else ts
    assert "pixels" in obs and "segmentation" in obs
    return {"pixels": obs["pixels"], "segmentation": obs["segmentation"]}

def _to_chw_stacked(obs_item: np.ndarray) -> torch.Tensor:
    """
    Accepts either:
      - [S, H, W, C]  -> returns [C*S, H, W]
      - [C, H, W]     -> returns as-is
    """
    if obs_item.ndim == 4:                            # [S,H,W,C]
        S, H, W, C = obs_item.shape
        x = torch.from_numpy(obs_item).permute(0, 3, 1, 2).contiguous()  # [S,C,H,W]
        return x.view(S * C, H, W)
    elif obs_item.ndim == 3:                          # [C,H,W]
        return torch.from_numpy(obs_item)
    else:
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
    seg_chw = _to_chw_stacked(obs["segmentation"])
    Ci, H, W = img_chw.shape
    Cs, H2, W2 = seg_chw.shape
    assert (H, W) == (H2, W2), "pixels/segmentation spatial size mismatch"
    return Ci, Cs, H, W

def collect_batch(env, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, segs = [], []
    ts = env.reset()
    for _ in range(batch_size):
        obs = _get_obs_dict(ts)
        imgs.append(_to_chw_stacked(obs["pixels"]))        # [Ci,H,W]
        segs.append(_to_chw_stacked(obs["segmentation"]))  # [Cs,H,W]
        ts = env.step(_sample_action(env))
        if hasattr(ts, "last") and ts.last():
            ts = env.reset()
    img = torch.stack(imgs, dim=0).float().to(device) / 255.0
    seg = torch.stack(segs, dim=0).float().to(device) / 255.0
    return img, seg

# --- mask + image helpers ----------------------------------------------------

def _patch_mask_from_indices(B, H, W, ph, pw, masked_indices, total_patches, device):
    """
    Build a [B, 1, H, W] binary mask where masked patches are 1.0.
    masked_indices: LongTensor [B, n_mask] over 0..(total_patches-1)
    ph,pw: patch size (height,width)
    """
    if total_patches == 0 or masked_indices.numel() == 0:
        return torch.zeros((B, 1, H, W), device=device)
    grid_h, grid_w = H // ph, W // pw
    cell = torch.zeros((B, 1, grid_h, grid_w), device=device)
    br = torch.arange(B, device=device)[:, None]
    mh = (masked_indices // grid_w)
    mw = (masked_indices %  grid_w)
    cell[br, 0, mh, mw] = 1.0
    mask = F.interpolate(cell, size=(H, W), mode="nearest")
    return mask  # [B,1,H,W]

def _overlay_mask(x, mask, alpha=0.65):
    """
    Darken masked areas on x (RGB or 1ch): x * (1 - alpha * mask)
    x: [B,C,H,W] in [0,1]; mask: [B,1,H,W] with {0,1}
    """
    return x * (1.0 - alpha * mask)

# ---------- reconstruction (no masking) ----------
@torch.no_grad()
def reconstruct_full(mae: VSMAE, img: torch.Tensor, seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build tokens without masking, run through encoder+decoder, and unpatchify.
    Robust to Early-CNN returning either [B,D,Hc,Wc] or [B,N,D].
    """
    device = next(mae.parameters()).device
    enc = mae.encoder
    B, Ci, H, W = img.shape
    _, Cs, _, _ = seg.shape

    # patch grids
    Hp = enc.image_height // enc.image_patch_height
    Wp = enc.image_width  // enc.image_patch_width
    Hs = enc.segmentation_height // enc.segmentation_patch_height
    Ws = enc.segmentation_width  // enc.segmentation_patch_width

    # ---- image tokens ----
    if Ci > 0:
        feat = mae.early_conv_vision(img)   # could be [B,D,Hc,Wc] or [B,N,D]
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, (Hp, Wp))  # [B,D,Hp,Wp]
            feat = feat.flatten(2).transpose(1, 2)        # [B,Hp*Wp,D]
        ti = feat
        if mae.use_sincosmod_encodings:
            ti = ti + mae.encoder_modality_embedding(torch.tensor(0, device=device)) + mae.image_enc_pos_embedding
    else:
        ti = torch.zeros((B, 0, mae.encoder_dim), device=device)

    # ---- segmentation tokens ----
    if Cs > 0:
        patches = enc.segmentation_to_patch_embedding[0](seg)
        ts = nn.Sequential(*enc.segmentation_to_patch_embedding[1:])(patches)
        if mae.use_sincosmod_encodings:
            ts = ts + mae.encoder_modality_embedding(torch.tensor(1, device=device)) + mae.segmentation_enc_pos_embedding
    else:
        ts = torch.zeros((B, 0, mae.encoder_dim), device=device)

    tokens = torch.cat([ti, ts], dim=1)                        # [B, Nt, D]
    Ni = ti.shape[1]

    enc_out = enc.transformer(tokens)
    dec_in = mae.enc_to_dec(enc_out)

    # add decoder modality + PE
    if mae.use_sincosmod_encodings:
        if Ni > 0:
            dec_in[:, :Ni] += mae.decoder_modality_embedding(torch.tensor(0, device=device)) + mae.image_dec_pos_embedding
        if ts.shape[1] > 0:
            dec_in[:, Ni:] += mae.decoder_modality_embedding(torch.tensor(1, device=device)) + mae.segmentation_dec_pos_embedding

    dec_out = mae.decoder(dec_in)
    pred_img_patches = mae.to_pixels(dec_out[:, :Ni]) if Ni > 0 else None
    pred_seg_patches = mae.to_segmentations(dec_out[:, Ni:]) if ts.shape[1] > 0 else None

    def unpatchify(patches, C, H, W, ph, pw):
        h = H // ph; w = W // pw
        return rearrange(patches, "b (h w) (ph pw c) -> b c (h ph) (w pw)", h=h, w=w, ph=ph, pw=pw, c=C)

    rec_img = torch.zeros_like(img)
    if pred_img_patches is not None:
        rec_img = unpatchify(pred_img_patches, enc.image_channels, enc.image_height, enc.image_width,
                             enc.image_patch_height, enc.image_patch_width).clamp(0, 1)

    rec_seg = torch.zeros_like(seg)
    if pred_seg_patches is not None:
        rec_seg = torch.sigmoid(pred_seg_patches)
        rec_seg = unpatchify(rec_seg, enc.segmentation_channels, enc.segmentation_height, enc.segmentation_width,
                             enc.segmentation_patch_height, enc.segmentation_patch_width).clamp(0, 1)

    return rec_img, rec_seg
def _draw_patch_boxes_single(
    img: torch.Tensor,
    masked_indices: torch.Tensor,
    ph: int,
    pw: int,
    total_patches: int,
    color=(0.5, 0.5, 0.5),  # grey overlay
    alpha: float = 0.4,     # transparency strength
) -> torch.Tensor:
    """
    img: [C,H,W] in [0,1], C>=3
    masked_indices: [n_mask] over 0..(total_patches-1)
    Apply a semi-transparent grey overlay on each masked patch.
    """
    if total_patches == 0 or masked_indices.numel() == 0:
        return img

    C, H, W = img.shape
    assert C >= 3, "Need at least 3 channels for RGB drawing"

    grid_h = H // ph
    grid_w = W // pw
    device = img.device

    # mask over H×W of where patches are masked
    patch_mask = torch.zeros((1, 1, H, W), device=device)

    idxs = masked_indices.view(-1).tolist()
    for idx in idxs:
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

    # grey overlay color
    color_tensor = torch.tensor(color, device=device).view(3, 1, 1)

    out = img.clone()
    out = out * (1.0 - alpha * mask) + color_tensor * (alpha * mask)

    return out


# ---------- save previews (first sample, first frame) ----------
@torch.no_grad()
def save_preview(
    step: int,
    out_dir: str,
    img_aug: torch.Tensor,       # [B,C,H,W] in [0,1]
    seg_aug: torch.Tensor,       # [B,Cs,H,W] in [0,1]
    rec_img: torch.Tensor,       # [B,C,H,W] in [0,1]
    rec_seg: torch.Tensor,       # [B,Cs,H,W] in [0,1]
    m_idx_img: torch.Tensor,     # [B, n_mask_img]
    m_idx_seg: torch.Tensor,     # [B, n_mask_seg]
    Ni: int,
    Ns: int,
    patch_h_img: int,
    patch_w_img: int,
    patch_h_seg: int,
    patch_w_seg: int,
    frame_rgb=3,
    frame_seg=1
):
    import torchvision.utils as vutils
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # pick sample 0
    b0_img = img_aug[0, :frame_rgb].clamp(0, 1)
    b0_seg = seg_aug[0, :frame_seg]
    r0_img = rec_img[0, :frame_rgb].clamp(0, 1)
    r0_seg = rec_seg[0, :frame_seg].clamp(0, 1)

    # per-image min-max for seg for visibility
    def norm01(x):
        mn = x.amin(dim=(1, 2), keepdim=True); mx = x.amax(dim=(1, 2), keepdim=True)
        return (x - mn) / (mx - mn + 1e-6)

    H, W = b0_img.shape[-2:]
    Hs, Ws = b0_seg.shape[-2:]

    device = b0_img.device
    B = img_aug.shape[0]

    # ---- RGB with blue boxes over masked patches ----
    if Ni > 0 and m_idx_img.numel() > 0:
        boxed_rgb = _draw_patch_boxes_single(
            b0_img, m_idx_img[0], patch_h_img, patch_w_img, Ni, color=(0.0, 0.0, 1.0)
        )
    else:
        boxed_rgb = b0_img

    # ---- Segmentation with blue boxes over masked patches ----
    # Make seg visualization 3-channel so we can use blue.
    # Use first seg channel for visualization.
    base_seg_1ch = b0_seg[0:1]
    seg_vis = base_seg_1ch.repeat(3, 1, 1)  # [3,Hs,Ws]
    if Ns > 0 and m_idx_seg.numel() > 0:
        boxed_seg = _draw_patch_boxes_single(
            seg_vis, m_idx_seg[0], patch_h_seg, patch_w_seg, Ns, color=(0.0, 0.0, 1.0)
        )
    else:
        boxed_seg = seg_vis

    # --- save separate files (sample 0) to disk (optional)
    vutils.save_image(b0_img,  os.path.join(out_dir, f"step_{step:06d}_aug_rgb.png"))
    vutils.save_image(norm01(b0_seg), os.path.join(out_dir, f"step_{step:06d}_aug_seg.png"))
    vutils.save_image(boxed_rgb,     os.path.join(out_dir, f"step_{step:06d}_masked_rgb.png"))
    vutils.save_image(boxed_seg,     os.path.join(out_dir, f"step_{step:06d}_masked_seg.png"))
    vutils.save_image(r0_img,        os.path.join(out_dir, f"step_{step:06d}_rec_rgb.png"))
    vutils.save_image(norm01(r0_seg), os.path.join(out_dir, f"step_{step:06d}_rec_seg.png"))

    # --- also log to WandB if active ---
    if wandb.run is not None:
        wandb.log(
            {
                "preview/aug_rgb": wandb.Image(b0_img.detach().cpu()),
                "preview/aug_seg": wandb.Image(norm01(b0_seg).detach().cpu()),
                "preview/masked_rgb": wandb.Image(boxed_rgb.detach().cpu()),
                "preview/masked_seg": wandb.Image(boxed_seg.detach().cpu()),
                "preview/rec_rgb": wandb.Image(r0_img.detach().cpu()),
                "preview/rec_seg": wandb.Image(norm01(r0_seg).detach().cpu()),
            },
            step=step,
        )

# --------------------------- main ---------------------------

def main():
    p = argparse.ArgumentParser()
    # Env
    p.add_argument("--env_name", type=str, default="walker_walk")
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
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
    p.add_argument("--steps", type=int, default=45_000)
    p.add_argument("--batch_size", type=int, default=32)  # was 128
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--decoder_lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--vis_every", type=int, default=10000)  # you can pass --vis_every 5000 for 5k previews
    p.add_argument("--out", type=str, default="vtmae_pretrained.pt")
    # WandB
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vtmae-only")
    p.add_argument("--wandb_run", type=str, default=None)

    # masking / seg loss
    p.add_argument("--masking_ratio_a", type=float, default=0.75)
    p.add_argument("--masking_ratio_b", type=float, default=0.75)
    p.add_argument("--seg_loss", type=float, default=0.25) # 1.0
    p.add_argument("--env_type", type=str, default="dmc")
    args = p.parse_args()

    out_images_dir = (str(args.masking_ratio_a) + "_" + str(args.masking_ratio_b) + "_" + str(args.seg_loss) + "_" + str(args.env_type))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.env_type == "dmc":

        # Choose environment
        env = make_env(
            name="walker_walk",
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            seed=args.seed,
            add_segmentation=True,
        )
    elif args.env_type == "mw":
        env = make_metaworld(
            name="button-press-topdown-v3",
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            seed=args.seed,
            camera_name="corner",
            add_segmentation_to_obs=True,
        )   

    # derive channels & size
    Ci, Cs, H, W = peek_channels(env)
    assert H % args.patch_size == 0 and W % args.patch_size == 0, \
        f"({H},{W}) must be divisible by patch_size={args.patch_size}"
    print(f"[VTMAE] image_channels={Ci}, segmentation_channels={Cs}, size=({H},{W})")

    v = VST(
        image_size=(H, W),
        segmentation_size=(H, W),
        image_patch_size=args.patch_size,
        segmentation_patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        image_channels=Ci,
        segmentation_channels=Cs,
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
        segloss_multiplier=args.seg_loss,
    ).to(device)

    enc_params = list(mae.encoder.parameters())
    if hasattr(mae, "early_conv_vision"):
        enc_params += list(mae.early_conv_vision.parameters())
    if hasattr(mae, "early_conv_segmentation"):
        enc_params += list(mae.early_conv_segmentation.parameters())
    encoder_opt = torch.optim.AdamW(enc_params, lr=args.encoder_lr, weight_decay=0.05)

    dec_params = (
        list(mae.decoder.parameters())
        + list(mae.to_pixels.parameters())
        + list(mae.to_segmentations.parameters())
        + list(mae.decoder_modality_embedding.parameters())
    )
    decoder_opt = torch.optim.AdamW(dec_params, lr=args.decoder_lr, weight_decay=0.0)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))


    mae.train()
    running = None
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        # batch
        img, seg = collect_batch(env, args.batch_size, device)      # [B,Ci,H,W], [B,Cs,H,W]
        assert img.shape[-1] == img.shape[-2] == seg.shape[-1] == seg.shape[-2], "RandomShiftsAug expects square"

        # DrQ-style aligned random shifts via grid_sample
        img_aug = img
        seg_aug = seg

        # VTMAE loss
        inputs = {"image": img_aug, "segmentation": seg_aug}
        out = mae(inputs, return_breakdown=True)

        recon_loss = out["total"]              # total loss used for backprop
        rgb_mse = out["rgb_mse"].detach().item()
        seg_bce = out["seg_bce"].detach().item()
        seg_weight = out["seg_weight"]         # float

        encoder_opt.zero_grad(set_to_none=True)
        decoder_opt.zero_grad(set_to_none=True)
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        encoder_opt.step()
        decoder_opt.step()

        # logs
        val = float(recon_loss.detach().cpu().item())
        running = val if running is None else (0.99 * running + 0.01 * val)

        if step % args.log_every == 0:
            print(
                f"[{step}/{args.steps}] "
                f"total={val:.4f} ema={running:.4f} "
                f"rgb_mse={rgb_mse:.4f} seg_bce={seg_bce:.4f} "
                f"seg_weight={seg_weight:.3f}",
                flush=True,
            )
            if args.wandb:
                wandb.log(
                    {
                        "mae_total": val,
                        "mae_total_ema": running,
                        "rgb_mse": rgb_mse,
                        "seg_bce": seg_bce,
                        "seg_weight": seg_weight,
                        # optional: how much of total is segmentation
                        "seg_contrib_frac": (seg_weight * seg_bce) / (val + 1e-8),
                    },
                    step=step,
                )



        if args.save_every and (step % args.save_every == 0):
            torch.save(mae.state_dict(), args.out)

        # previews
        if step % args.vis_every == 0:
            mae.eval()
            with torch.no_grad():
                # 1) get real masking indices by calling forward in debug mode (no grad)
                loss_dbg, dbg = mae({"image": img_aug, "segmentation": seg_aug}, return_debug=True)
                m_idx_img = dbg["m_idx_img"]        # [B, n_mask_img]
                m_idx_seg = dbg["m_idx_seg"]        # [B, n_mask_seg]
                Ni = dbg["Ni"]                      # number of image tokens (Hp*Wp)
                Ns = dbg["Ns"]                      # number of seg tokens (Hs*Ws)

                # 2) full reconstructions (no masking) for nice visuals
                rec_img, rec_seg = reconstruct_full(mae, img_aug, seg_aug)
            mae.train()

            # modality-specific patch sizes (usually equal, but kept separate)
            patch_h_img = mae.encoder.image_patch_height
            patch_w_img = mae.encoder.image_patch_width
            patch_h_seg = mae.encoder.segmentation_patch_height
            patch_w_seg = mae.encoder.segmentation_patch_width

            frame_rgb = 3
            frame_seg = max(1, Cs // args.frame_stack)  # usually 1

            save_preview(
                step, out_images_dir,
                img_aug, seg_aug, rec_img, rec_seg,
                m_idx_img=m_idx_img, m_idx_seg=m_idx_seg,
                Ni=Ni, Ns=Ns,
                patch_h_img=patch_h_img, patch_w_img=patch_w_img,
                patch_h_seg=patch_h_seg, patch_w_seg=patch_w_seg,
                frame_rgb=frame_rgb, frame_seg=frame_seg
            )

    torch.save(mae.state_dict(), args.out)
    if args.wandb:
        wandb.save(args.out)
    print(f"Done. Saved weights to {args.out}")

if __name__ == "__main__":
    main()


"""
sample run:

pretrain_vtmae.py --camera_main topview --camera_aux corner --frame_stack 3 
--action_repeat 2 --patch_size 6 --masking_ratio_a 1.0 --masking_ratio_b 0.75 
--aux_loss 1.0 --batch_size 32 --wandb --wandb_project vtmae-only --wandb_run mw_vitonly_topview_corner


"""