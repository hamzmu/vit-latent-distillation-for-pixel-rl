from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw

from metaworld_dm_env import make_metaworld


def to_hwc(obs_item: np.ndarray) -> np.ndarray:
    if obs_item.ndim == 3 and obs_item.shape[0] in (1, 3, 4):
        return np.transpose(obs_item[:3], (1, 2, 0))
    if obs_item.ndim == 3:
        return obs_item
    if obs_item.ndim == 4:
        first = obs_item[0]
        if first.ndim == 3 and first.shape[0] in (1, 3, 4):
            return np.transpose(first[:3], (1, 2, 0))
        return first
    raise ValueError(f"Unexpected obs shape {obs_item.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default="joint_mad_camera_views.png")
    args = p.parse_args()

    camera_names = ["gripperPOV", "corner2", "corner3"]
    alias_names = ["first", "third1", "third2"]
    env = make_metaworld(
        name="button-press-topdown-v3",
        frame_stack=1,
        action_repeat=2,
        seed=args.seed,
        proprio=False,
        camera_names=camera_names,
    )
    try:
        ts = env.reset()
        obs = ts.observation
        keys = ["pixels", "pixels_aux", "pixels_aux2"]
        imgs = [to_hwc(obs[k]) for k in keys]
    finally:
        try:
            env.close()
        except Exception:
            pass

    tile_h = max(img.shape[0] for img in imgs)
    tile_w = max(img.shape[1] for img in imgs)
    row = []
    for idx, img in enumerate(imgs):
        tile = Image.new("RGB", (tile_w, tile_h + 20), color=(18, 18, 18))
        tile_img = Image.fromarray(img)
        tile.paste(tile_img, (0, 20))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 2), f"cam{idx}: {alias_names[idx]} -> {camera_names[idx]}", fill=(235, 235, 235))
        row.append(np.array(tile))

    canvas = Image.fromarray(np.concatenate(row, axis=1))
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f"Saved image: {out_path}")
    for idx in range(3):
        print(f"cam{idx}: {alias_names[idx]} -> {camera_names[idx]}")


if __name__ == "__main__":
    main()
