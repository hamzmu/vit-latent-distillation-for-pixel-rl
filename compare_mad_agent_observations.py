from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parent
MAD_DIR = REPO_ROOT / "mad"
if str(MAD_DIR) not in sys.path:
    sys.path.append(str(MAD_DIR))

from camera_aliases import normalize_camera_names, resolve_camera_alias_profile
import envs


def chw_to_hwc(chw: np.ndarray) -> np.ndarray:
    assert chw.ndim == 3, f"Expected [C,H,W], got {chw.shape}"
    return np.transpose(chw[:3], (1, 2, 0))


def pad_tile(img: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    out[:h, :w] = img
    return out


def capture_observation_for_profile(agent_name: str, seed: int) -> tuple[np.ndarray, list[str]]:
    cfg = OmegaConf.load(MAD_DIR / "config.yaml")
    cfg.task = "button-press-topdown"
    cfg.agent = agent_name
    cfg.device = "cpu"
    cfg.metaworld_backend = "ours"
    cfg.seed = seed

    env = envs.make(cfg)
    try:
        ts = env.reset()
        obs = np.asarray(ts.observation)
    finally:
        try:
            env.close()
        except Exception:
            pass

    profile = resolve_camera_alias_profile(cfg.agent, cfg.camera_alias_profile)
    resolved = normalize_camera_names(list(cfg.cameras), profile)
    return obs, resolved


def capture_training_observation(seed: int) -> tuple[np.ndarray, list[str]]:
    cfg = OmegaConf.load(MAD_DIR / "config.yaml")
    cfg.task = "button-press-topdown"
    cfg.agent = "mad"
    cfg.device = "cpu"
    cfg.metaworld_backend = "ours"
    cfg.camera_alias_profile = "mad"
    cfg.seed = seed

    env = envs.make(cfg)
    try:
        ts = env.reset()
        obs = np.asarray(ts.observation)
    finally:
        try:
            env.close()
        except Exception:
            pass

    resolved = normalize_camera_names(list(cfg.cameras), "mad")
    return obs, resolved


def row_image(obs: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    tiles = []
    for vidx in range(obs.shape[0]):
        tile = Image.fromarray(pad_tile(chw_to_hwc(obs[vidx]), tile_h, tile_w))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_w, 18), fill=(0, 0, 0))
        draw.text((4, 3), f"cam{vidx}", fill=(255, 255, 255))
        tiles.append(np.array(tile))
    return np.concatenate(tiles, axis=1)


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
    p.add_argument("--out", type=str, default="mad_three_agent_observations.png")
    args = p.parse_args()

    cache: dict[tuple[str, ...], np.ndarray] = {}
    captures = []
    train_obs, train_resolved = capture_training_observation(args.seed)
    cache[tuple(train_resolved)] = train_obs.copy()
    captures.append(("joint_madcams_vit_training", train_obs, train_resolved))
    for agent_name in ("mad", "vit_latent_full_sac", "vit_latent_subset_sac"):
        obs, resolved = capture_observation_for_profile(agent_name, args.seed)
        key = tuple(resolved)
        if key in cache:
            obs = cache[key].copy()
        else:
            cache[key] = obs.copy()
        captures.append((agent_name, obs, resolved))

    tile_h = max(obs.shape[2] for _, obs, _ in captures)
    tile_w = max(obs.shape[3] for _, obs, _ in captures)
    rows = [
        (
            f"{agent_name}: ['first', 'third1', 'third2'] -> {resolved}",
            row_image(obs, tile_h, tile_w),
        )
        for agent_name, obs, resolved in captures
    ]
    canvas = build_canvas(rows)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    labels = [f"{agent_name}: ['first', 'third1', 'third2'] -> {resolved}" for agent_name, _, resolved in captures]
    obs_arrays = [obs for _, obs, _ in captures]
    print(f"Saved comparison image: {out_path}")
    for label in labels:
        print(label)
    print(f"mad == vit_latent_full_sac: {np.array_equal(obs_arrays[0], obs_arrays[1])}")
    print(f"mad == vit_latent_subset_sac: {np.array_equal(obs_arrays[0], obs_arrays[2])}")
    print(f"vit_latent_full_sac == vit_latent_subset_sac: {np.array_equal(obs_arrays[1], obs_arrays[2])}")


if __name__ == "__main__":
    main()
