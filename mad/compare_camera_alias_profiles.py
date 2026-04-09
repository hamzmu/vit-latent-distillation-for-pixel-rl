from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

from camera_aliases import describe_camera_mapping
from envs.metaworld import make


def chw_to_hwc(obs: np.ndarray) -> np.ndarray:
    assert obs.ndim == 3, f"Expected [C,H,W], got {obs.shape}"
    return np.transpose(obs, (1, 2, 0))


def pad_tile(img: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    out[:h, :w] = img
    return out


def render_row(task: str, cameras: list[str], profile: str, img_size: int, seed: int) -> tuple[str, np.ndarray]:
    cfg = OmegaConf.create(
        {
            "task": task,
            "agent": "vit_latent_full_sac" if profile == "vit_joint" else "mad",
            "camera_alias_profile": profile,
            "cameras": cameras,
            "frame_stack": 1,
            "seed": seed,
            "img_size": img_size,
        }
    )
    env = make(cfg)
    try:
        ts = env.reset()
        obs = ts.observation
        tiles = [chw_to_hwc(obs[i, :3]) for i in range(obs.shape[0])]
    finally:
        try:
            env.close()
        except Exception:
            pass

    mappings = describe_camera_mapping(cameras, profile)
    title = " | ".join(f"{src}->{dst}" for src, dst in mappings)
    return title, np.concatenate(tiles, axis=1)


def build_canvas(rows: list[tuple[str, np.ndarray]]) -> Image.Image:
    row_height = max(img.shape[0] for _, img in rows)
    row_width = max(img.shape[1] for _, img in rows)
    title_height = 28
    gap = 10
    canvas_height = len(rows) * (row_height + title_height + gap) + gap
    canvas_width = row_width + 2 * gap

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
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
    p.add_argument("--task", type=str, default="button-press-topdown")
    p.add_argument("--img_size", type=int, default=84)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out",
        type=str,
        default="camera_alias_profile_comparison_button_press_topdown.png",
    )
    args = p.parse_args()

    rows = [
        render_row(args.task, ["gripperPOV", "corner", "corner2"], "vit_joint", args.img_size, args.seed),
        render_row(args.task, ["first", "third1", "third2"], "mad", args.img_size, args.seed),
        render_row(args.task, ["first", "third1", "third2"], "vit_joint", args.img_size, args.seed),
    ]
    canvas = build_canvas(rows)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f"Saved comparison image: {out_path}")
    for title, _ in rows:
        print(title)


if __name__ == "__main__":
    main()
