from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
MAD_DIR = REPO_ROOT / "mad"
if str(MAD_DIR) not in sys.path:
    sys.path.append(str(MAD_DIR))

from camera_aliases import describe_camera_mapping
from envs.metaworld import make as make_mad_metaworld
from metaworld_dm_env import make_metaworld as make_our_metaworld


def chw_to_hwc(obs: np.ndarray) -> np.ndarray:
    assert obs.ndim == 3, f"Expected [C,H,W], got {obs.shape}"
    return np.transpose(obs, (1, 2, 0))


def to_hwc_from_our_obs(obs_item: np.ndarray) -> np.ndarray:
    """
    Our wrapper can return:
    - [S, H, W, C]
    - [S, C, H, W]
    - [H, W, C]
    - [C, H, W]
    """
    if obs_item.ndim == 4:
        first = obs_item[0]
        if first.ndim == 3 and first.shape[0] in (1, 3, 4):
            return np.transpose(first, (1, 2, 0))
        return first
    if obs_item.ndim == 3:
        if obs_item.shape[0] in (1, 3, 4):
            return np.transpose(obs_item, (1, 2, 0))
        return obs_item
    raise ValueError(f"Unexpected our-wrapper obs shape: {obs_item.shape}")


def pad_tile(img: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    out[:h, :w] = img
    return out


def seeded(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def capture_our_views(seed: int) -> tuple[str, list[np.ndarray]]:
    seeded(seed)
    camera_names = ["gripperPOV", "corner", "corner2"]
    env = make_our_metaworld(
        name="button-press-topdown-v3",
        frame_stack=1,
        action_repeat=2,
        seed=seed,
        camera_names=camera_names,
    )
    try:
        ts = env.reset()
        obs = ts.observation
        keys = ["pixels", "pixels_aux", "pixels_aux2"]
        imgs = [to_hwc_from_our_obs(obs[k]) for k in keys]
    finally:
        try:
            env.close()
        except Exception:
            pass
    title = "Our pretrain env: cam0=gripperPOV | cam1=corner | cam2=corner2"
    return title, imgs


def capture_mad_views(seed: int, profile: str) -> tuple[str, list[np.ndarray]]:
    seeded(seed)
    cfg = OmegaConf.create(
        {
            "task": "button-press-topdown",
            "agent": "vit_latent_full_sac" if profile == "vit_joint" else "mad",
            "camera_alias_profile": profile,
            "cameras": ["first", "third1", "third2"],
            "frame_stack": 1,
            "seed": seed,
            "img_size": 84,
        }
    )
    env = make_mad_metaworld(cfg)
    try:
        ts = env.reset()
        obs = ts.observation
        imgs = [chw_to_hwc(obs[i, :3]) for i in range(obs.shape[0])]
    finally:
        try:
            env.close()
        except Exception:
            pass
    mapping = " | ".join(f"{src}->{dst}" for src, dst in describe_camera_mapping(cfg.cameras, profile))
    title = f"MAD button-press-topdown ({profile}): {mapping}"
    return title, imgs


def make_row(label: str, imgs: list[np.ndarray], tile_w: int, tile_h: int) -> tuple[str, np.ndarray]:
    labeled_tiles = []
    for idx, img in enumerate(imgs):
        tile = Image.fromarray(pad_tile(img, tile_h, tile_w))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_w, 18), fill=(0, 0, 0))
        draw.text((4, 3), f"cam{idx}", fill=(255, 255, 255))
        labeled_tiles.append(np.array(tile))
    return label, np.concatenate(labeled_tiles, axis=1)


def build_canvas(rows: list[tuple[str, np.ndarray]]) -> Image.Image:
    row_height = max(img.shape[0] for _, img in rows)
    row_width = max(img.shape[1] for _, img in rows)
    title_height = 30
    gap = 10
    canvas_height = len(rows) * (row_height + title_height + gap) + gap
    canvas_width = row_width + 2 * gap

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    y = gap
    for title, img in rows:
        draw.text((gap, y), title, fill=(235, 235, 235))
        y += title_height
        tile = Image.fromarray(pad_tile(img, row_height, row_width))
        canvas.paste(tile, (gap, y))
        y += row_height + gap
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out",
        type=str,
        default="our_vs_mad_button_press_observations.png",
    )
    args = p.parse_args()

    our_title, our_imgs = capture_our_views(args.seed)
    mad_default_title, mad_default_imgs = capture_mad_views(args.seed, "mad")
    mad_vit_title, mad_vit_imgs = capture_mad_views(args.seed, "vit_joint")

    tile_h = max(img.shape[0] for img in our_imgs + mad_default_imgs + mad_vit_imgs)
    tile_w = max(img.shape[1] for img in our_imgs + mad_default_imgs + mad_vit_imgs)

    rows = [
        make_row(our_title, our_imgs, tile_w, tile_h),
        make_row(mad_default_title, mad_default_imgs, tile_w, tile_h),
        make_row(mad_vit_title, mad_vit_imgs, tile_w, tile_h),
    ]
    canvas = build_canvas(rows)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f"Saved comparison image: {out_path}")
    print(our_title)
    print(mad_default_title)
    print(mad_vit_title)


if __name__ == "__main__":
    main()
