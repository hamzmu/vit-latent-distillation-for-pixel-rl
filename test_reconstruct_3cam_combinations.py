from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.pop("DISPLAY", None)

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from metaworld_dm_env import make_metaworld
from pretrain_models import MultiViewVST, MultiViewVSMAE


def pixel_obs_key(view_idx: int) -> str:
    if view_idx == 0:
        return "pixels"
    if view_idx == 1:
        return "pixels_aux"
    return f"pixels_aux{view_idx}"


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


@torch.no_grad()
def _sample_action(env) -> np.ndarray:
    spec = env.action_spec()
    shape = spec.shape if getattr(spec, "shape", ()) else (1,)
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)


@torch.no_grad()
def _get_obs_dict(ts, view_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    obs = ts.observation if hasattr(ts, "observation") else ts
    missing = [k for k in view_keys if k not in obs]
    assert not missing, f"Missing view keys {missing}; available={list(obs.keys())}"
    return {k: obs[k] for k in view_keys}


def _to_chw_stacked(obs_item: np.ndarray) -> torch.Tensor:
    if obs_item.ndim == 4:  # [S,H,W,C]
        s, h, w, c = obs_item.shape
        x = torch.from_numpy(obs_item).permute(0, 3, 1, 2).contiguous()
        return x.view(s * c, h, w)
    if obs_item.ndim == 3:
        if obs_item.shape[-1] in (1, 3, 4):
            return torch.from_numpy(obs_item).permute(2, 0, 1).contiguous()
        return torch.from_numpy(obs_item)
    raise AssertionError(f"Unexpected obs shape {obs_item.shape}")


def capture_views(env, view_keys: Sequence[str], device: torch.device, random_steps: int) -> torch.Tensor:
    ts = env.reset()
    for _ in range(max(0, int(random_steps))):
        ts = env.step(_sample_action(env))
        if hasattr(ts, "last") and ts.last():
            ts = env.reset()

    obs = _get_obs_dict(ts, view_keys)
    per_view = [_to_chw_stacked(obs[k]) for k in view_keys]
    views = torch.stack(per_view, dim=0).unsqueeze(0).float().to(device) / 255.0  # [1,V,C,H,W]
    return views


def unpatchify(patches: torch.Tensor, C: int, H: int, W: int, ph: int, pw: int) -> torch.Tensor:
    b, n, d = patches.shape
    h = H // ph
    w = W // pw
    assert n == h * w, f"Patch count mismatch: n={n}, expected={h*w}"
    assert d == ph * pw * C, f"Patch dim mismatch: d={d}, expected={ph*pw*C}"

    x = patches.view(b, h, w, ph, pw, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, C, H, W)


def reconstruct_from_debug(mae: MultiViewVSMAE, views: torch.Tensor, debug: dict) -> torch.Tensor:
    """
    Build composite reconstructions:
      - unmasked patches from target
      - masked patches from predictions
    Returns [B,V,C,H,W]
    """
    target_patches = debug["target_patches"]  # [B,V,N,P]
    pred_patches = debug["pred_patches"]      # [B,V,N,P]
    loss_mask = debug["loss_mask"]            # [B,V,N]

    recon_patches = target_patches.clone()
    recon_patches[loss_mask] = pred_patches[loss_mask]

    _, V, _, _ = recon_patches.shape
    _, _, C, H, W = views.shape

    recon_views = []
    for vidx in range(V):
        recon_views.append(
            unpatchify(
                recon_patches[:, vidx],
                C=C,
                H=H,
                W=W,
                ph=mae.patch_height,
                pw=mae.patch_width,
            )
        )
    return torch.stack(recon_views, dim=1).clamp(0, 1)


def _latest_rgb(x_chw: torch.Tensor, frame_rgb: int) -> torch.Tensor:
    c, _, _ = x_chw.shape
    assert c >= frame_rgb, f"Expected at least {frame_rgb} channels, got {c}"
    return x_chw[-frame_rgb:].clamp(0, 1)


def _to_pil(x_chw: torch.Tensor) -> Image.Image:
    arr = (x_chw.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_combo_grid(
    out_path: Path,
    combo_title: str,
    camera_names: Sequence[str],
    orig_views: torch.Tensor,   # [V,C,H,W]
    recon_views: torch.Tensor,  # [V,C,H,W]
    frame_rgb: int,
):
    V = orig_views.shape[0]
    assert V == len(camera_names), f"camera_names length {len(camera_names)} != V={V}"

    orig_imgs = [_to_pil(_latest_rgb(orig_views[v], frame_rgb)) for v in range(V)]
    recon_imgs = [_to_pil(_latest_rgb(recon_views[v], frame_rgb)) for v in range(V)]

    w, h = orig_imgs[0].size
    pad = 12
    top_h = 30
    label_h = 16
    rows = 2
    cols = V

    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = top_h + rows * (h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 22))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, 8), combo_title, fill=(235, 235, 235))

    for col in range(cols):
        x0 = pad + col * (w + pad)

        y0 = top_h + pad
        canvas.paste(orig_imgs[col], (x0, y0 + label_h))
        draw.text((x0, y0), f"orig {camera_names[col]}", fill=(220, 220, 220))

        y1 = top_h + 2 * pad + h + label_h
        canvas.paste(recon_imgs[col], (x0, y1 + label_h))
        draw.text((x0, y1), f"recon {camera_names[col]}", fill=(220, 220, 220))

    canvas.save(out_path)


def all_nonempty_view_subsets(num_views: int = 3) -> List[List[bool]]:
    subsets: List[List[bool]] = []
    for mask in range(1, 1 << num_views):
        subset = [bool(mask & (1 << i)) for i in range(num_views)]
        subsets.append(subset)
    # Sort by number of visible cams first: 1-cam, 2-cam, 3-cam
    subsets.sort(key=lambda x: (sum(x), x))
    return subsets


def full_and_single_view_subsets(num_views: int = 3) -> List[List[bool]]:
    assert num_views == 3, f"Expected num_views=3, got {num_views}"
    return [
        [True, True, True],
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]


def subset_name(subset: Sequence[bool]) -> str:
    cams = [f"cam{i}" for i, keep in enumerate(subset) if keep]
    return "keep_" + "_".join(cams)


def pair_name(a: str, b: str) -> str:
    return f"{a}__vs__{b}"


def save_method_comparison_grid(
    out_path: Path,
    combo_title: str,
    camera_names: Sequence[str],
    orig_views: torch.Tensor,  # [V,C,H,W]
    method_recons: Sequence[Tuple[str, torch.Tensor]],  # [(label, [V,C,H,W])]
    frame_rgb: int,
):
    V = orig_views.shape[0]
    assert V == len(camera_names), f"camera_names length {len(camera_names)} != V={V}"

    orig_imgs = [_to_pil(_latest_rgb(orig_views[v], frame_rgb)) for v in range(V)]
    recon_rows = [(label, [_to_pil(_latest_rgb(recon[v], frame_rgb)) for v in range(V)]) for label, recon in method_recons]

    w, h = orig_imgs[0].size
    pad = 12
    top_h = 34
    label_w = 120
    label_h = 18
    rows = 1 + len(recon_rows)
    cols = V

    canvas_w = label_w + cols * w + (cols + 2) * pad
    canvas_h = top_h + rows * (h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), combo_title, fill=(235, 235, 235))

    for row_idx in range(rows):
        if row_idx == 0:
            row_label = "orig"
            row_imgs = orig_imgs
        else:
            row_label, row_imgs = recon_rows[row_idx - 1]

        y0 = top_h + pad + row_idx * (h + label_h + pad)
        draw.text((pad, y0 + label_h + h // 2 - 6), row_label, fill=(220, 220, 220))

        for col_idx in range(cols):
            x0 = label_w + (col_idx + 1) * pad + col_idx * w
            if row_idx == 0:
                draw.text((x0, y0), camera_names[col_idx], fill=(220, 220, 220))
            canvas.paste(row_imgs[col_idx], (x0, y0 + label_h))

    canvas.save(out_path)


def resolve_checkpoints(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    provided = int(args.checkpoint is not None) + int(bool(args.checkpoints)) + int(args.compare_seq_tag is not None)
    assert provided == 1, "Provide exactly one of --checkpoint, --checkpoints, or --compare_seq_tag."

    if args.compare_seq_tag is not None:
        seq_root = Path(args.seq_runs_dir)
        tag = str(args.compare_seq_tag)
        methods = [
            ("joint", seq_root / f"joint_{tag}"),
            ("distill", seq_root / f"distill_{tag}"),
            ("curriculum", seq_root / f"curriculum_{tag}"),
        ]
        resolved = []
        for label, method_dir in methods:
            matches = sorted(method_dir.glob("*.pt"))
            assert matches, f"No checkpoint found for {label} in {method_dir}"
            assert len(matches) == 1, f"Expected one checkpoint in {method_dir}, found {matches}"
            resolved.append((label, matches[0]))
        return resolved

    if args.checkpoints:
        paths = [Path(p) for p in args.checkpoints]
        labels = list(args.checkpoint_labels) if args.checkpoint_labels else [p.stem for p in paths]
        assert len(labels) == len(paths), "checkpoint_labels length must match checkpoints length"
        return list(zip(labels, paths))

    assert args.checkpoint is not None
    checkpoint = Path(args.checkpoint)
    label = args.checkpoint_label or checkpoint.stem
    return [(label, checkpoint)]


def build_mae(args: argparse.Namespace, *, channels: int, height: int, width: int, device: torch.device) -> MultiViewVSMAE:
    encoder = MultiViewVST(
        image_size=(height, width),
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        channels=channels,
        num_views=3,
        frame_stack=args.frame_stack,
    )
    mae = MultiViewVSMAE(
        encoder=encoder,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        num_views=3,
        frame_stack=args.frame_stack,
    ).to(device)
    return mae


def evaluate_checkpoint(
    *,
    label: str,
    checkpoint_path: Path,
    mae: MultiViewVSMAE,
    views: torch.Tensor,
    subsets: Sequence[Sequence[bool]],
    args: argparse.Namespace,
    camera_names: Sequence[str],
    model_out_dir: Path,
    device: torch.device,
) -> dict:
    state = torch.load(checkpoint_path, map_location=device)
    mae.load_state_dict(state, strict=True)
    mae.eval()

    print(f"[Eval:{label}] checkpoint={checkpoint_path}")
    print(f"[Eval:{label}] output_dir={model_out_dir.resolve()}")

    subset_latents: Dict[str, torch.Tensor] = {}
    subset_metrics: Dict[str, Dict[str, float | str]] = {}
    subset_recons: Dict[str, torch.Tensor] = {}
    summary_lines: List[str] = []

    full_subset_tensor = torch.tensor([[True, True, True]], device=device, dtype=torch.bool)
    with torch.no_grad():
        z_full = mae.get_embeddings(views, visible_views=full_subset_tensor, eval=True)

    for subset in subsets:
        subset_tensor = torch.tensor([subset], device=device, dtype=torch.bool)
        visible_ratio = float(max(0.0, min(1.0, args.visible_mask_ratio)))
        mask_ratios = torch.where(
            subset_tensor,
            torch.full((1, 3), visible_ratio, device=device, dtype=views.dtype),
            torch.ones((1, 3), device=device, dtype=views.dtype),
        )

        with torch.no_grad():
            out = mae(
                views,
                visible_views=subset_tensor,
                mask_ratios=mask_ratios,
                cross_view_loss_weight=args.cross_view_loss_weight,
                return_breakdown=True,
                return_debug=True,
            )
            recon_views = reconstruct_from_debug(mae, views, out["debug"])
            z_subset = mae.get_embeddings(views, visible_views=subset_tensor, eval=True)

        combo = subset_name(subset)
        subset_latents[combo] = z_subset.detach().cpu()
        subset_recons[combo] = recon_views[0].detach().cpu()
        combo_title = f"{label} | {combo} | visible={subset} | visible_mask_ratio={visible_ratio:.2f}"
        combo_dir = model_out_dir / combo
        combo_dir.mkdir(parents=True, exist_ok=True)

        grid_path = combo_dir / f"{combo}_orig_vs_recon.png"
        save_combo_grid(
            out_path=grid_path,
            combo_title=combo_title,
            camera_names=camera_names,
            orig_views=views[0],
            recon_views=recon_views[0],
            frame_rgb=args.frame_rgb,
        )

        for vidx, cam_name in enumerate(camera_names):
            recon_img = _to_pil(_latest_rgb(recon_views[0, vidx], args.frame_rgb))
            recon_img.save(combo_dir / f"{combo}_recon_view{vidx}_{_sanitize(cam_name)}.png")

        z_cos_to_full = float(F.cosine_similarity(z_subset, z_full, dim=-1).mean().item())
        z_mse_to_full = float(F.mse_loss(z_subset, z_full).item())
        metrics = {
            "total": float(out["total"]),
            "v0": float(out["view0_mse"]),
            "v1": float(out["view1_mse"]),
            "v2": float(out["view2_mse"]),
            "z_cos_to_full": z_cos_to_full,
            "z_mse_to_full": z_mse_to_full,
            "grid": grid_path.name,
        }
        subset_metrics[combo] = metrics
        line = (
            f"{combo}: total={metrics['total']:.6f} "
            f"v0={metrics['v0']:.6f} "
            f"v1={metrics['v1']:.6f} "
            f"v2={metrics['v2']:.6f} "
            f"z_cos_to_full={metrics['z_cos_to_full']:.6f} "
            f"z_mse_to_full={metrics['z_mse_to_full']:.6f} "
            f"grid={metrics['grid']}"
        )
        print(f"[Eval:{label}] {line}")
        summary_lines.append(line)

    summary_path = model_out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    latent_lines: List[str] = []
    subset_names = list(subset_latents.keys())
    for i, name_a in enumerate(subset_names):
        z_a = subset_latents[name_a]
        for j, name_b in enumerate(subset_names):
            if j <= i:
                continue
            z_b = subset_latents[name_b]
            latent_lines.append(
                f"{pair_name(name_a, name_b)}: "
                f"cosine={float(F.cosine_similarity(z_a, z_b, dim=-1).mean().item()):.6f} "
                f"mse={float(F.mse_loss(z_a, z_b).item()):.6f}"
            )
    latent_path = model_out_dir / "latent_similarity.txt"
    latent_path.write_text("\n".join(latent_lines) + ("\n" if latent_lines else ""), encoding="utf-8")
    print(f"[Done:{label}] summary={summary_path}")
    print(f"[Done:{label}] latent={latent_path}")

    return {
        "label": label,
        "checkpoint": checkpoint_path,
        "subset_metrics": subset_metrics,
        "subset_recons": subset_recons,
        "subset_latents": subset_latents,
        "summary_path": summary_path,
    }


def main():
    p = argparse.ArgumentParser(description="Test reconstruction for all 3-camera subsets.")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to one trained 3-cam MAE checkpoint.")
    p.add_argument("--checkpoint_label", type=str, default=None, help="Optional label for --checkpoint.")
    p.add_argument("--checkpoints", nargs="+", type=str, default=None, help="Multiple checkpoints to compare.")
    p.add_argument("--checkpoint_labels", nargs="+", type=str, default=None, help="Labels matching --checkpoints.")
    p.add_argument("--compare_seq_tag", type=str, default=None, help="Compare joint/distill/curriculum checkpoints from seq_runs using this timestamp tag.")
    p.add_argument("--seq_runs_dir", type=str, default="seq_runs", help="Root directory containing seq_runs outputs.")
    p.add_argument("--output_dir", type=str, default="recon_3cam_combinations")
    p.add_argument("--env_name", type=str, default="button-press-topdown-v3")
    p.add_argument("--camera_names", nargs="+", type=str, default=["gripperPOV", "corner", "corner2"])
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env_random_steps", type=int, default=6)
    p.add_argument("--frame_rgb", type=int, default=3)
    p.add_argument("--visible_mask_ratio", type=float, default=0.0, help="Patch mask ratio for visible cameras during test.")
    p.add_argument("--cross_view_loss_weight", type=float, default=1.5)
    p.add_argument(
        "--subset_mode",
        type=str,
        default="full_and_singles",
        choices=["full_and_singles", "all_nonempty"],
        help="Which camera subsets to evaluate.",
    )

    # Must match training config of checkpoint.
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--decoder_dim", type=int, default=256)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--decoder_heads", type=int, default=4)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = resolve_checkpoints(args)

    env = make_metaworld(
        name=args.env_name,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        camera_names=camera_names,
        add_aux_pixels_to_obs=True,
    )
    view_keys = [pixel_obs_key(i) for i in range(3)]
    views = capture_views(env, view_keys, device=device, random_steps=args.env_random_steps)  # [1,3,C,H,W]

    _, V, C, H, W = views.shape
    assert V == 3, f"Expected 3 views, got {V}"
    assert H % args.patch_size == 0 and W % args.patch_size == 0, (
        f"Image size {(H, W)} must be divisible by patch_size={args.patch_size}"
    )
    print(f"[Eval] cameras={camera_names}")
    print(f"[Eval] output_dir={out_dir.resolve()}")
    print(f"[Eval] models={[label for label, _ in checkpoints]}")
    print("")

    if args.subset_mode == "full_and_singles":
        subsets = full_and_single_view_subsets(3)
    else:
        subsets = all_nonempty_view_subsets(3)

    all_results: List[dict] = []
    for label, checkpoint_path in checkpoints:
        mae = build_mae(args, channels=C, height=H, width=W, device=device)
        model_out_dir = out_dir / _sanitize(label)
        model_out_dir.mkdir(parents=True, exist_ok=True)
        all_results.append(
            evaluate_checkpoint(
                label=label,
                checkpoint_path=checkpoint_path,
                mae=mae,
                views=views,
                subsets=subsets,
                args=args,
                camera_names=camera_names,
                model_out_dir=model_out_dir,
                device=device,
            )
        )
        print("")

    comparison_lines: List[str] = []
    comparison_csv_lines: List[str] = ["subset,method,total,v0,v1,v2,z_cos_to_full,z_mse_to_full,checkpoint"]
    for subset in subsets:
        combo = subset_name(subset)
        comparison_lines.append(f"[{combo}]")
        ranked = sorted(
            [
                (result["label"], result["subset_metrics"][combo], result["checkpoint"])
                for result in all_results
            ],
            key=lambda item: float(item[1]["total"]),
        )
        for label, metrics, checkpoint_path in ranked:
            comparison_lines.append(
                f"{label}: total={float(metrics['total']):.6f} "
                f"v0={float(metrics['v0']):.6f} "
                f"v1={float(metrics['v1']):.6f} "
                f"v2={float(metrics['v2']):.6f} "
                f"z_cos_to_full={float(metrics['z_cos_to_full']):.6f} "
                f"z_mse_to_full={float(metrics['z_mse_to_full']):.6f}"
            )
            comparison_csv_lines.append(
                f"{combo},{label},{float(metrics['total']):.6f},{float(metrics['v0']):.6f},"
                f"{float(metrics['v1']):.6f},{float(metrics['v2']):.6f},"
                f"{float(metrics['z_cos_to_full']):.6f},{float(metrics['z_mse_to_full']):.6f},"
                f"{checkpoint_path}"
            )
        comparison_lines.append("")

        if len(all_results) > 1:
            save_method_comparison_grid(
                out_path=out_dir / f"{combo}_compare_methods.png",
                combo_title=f"{combo} | method comparison",
                camera_names=camera_names,
                orig_views=views[0],
                method_recons=[(result["label"], result["subset_recons"][combo]) for result in all_results],
                frame_rgb=args.frame_rgb,
            )

    comparison_summary_path = out_dir / "comparison_summary.txt"
    comparison_summary_path.write_text("\n".join(comparison_lines).rstrip() + "\n", encoding="utf-8")
    comparison_csv_path = out_dir / "comparison_summary.csv"
    comparison_csv_path.write_text("\n".join(comparison_csv_lines) + "\n", encoding="utf-8")
    print(f"[Done] Wrote comparison summary: {comparison_summary_path}")
    print(f"[Done] Wrote comparison csv: {comparison_csv_path}")


if __name__ == "__main__":
    main()
