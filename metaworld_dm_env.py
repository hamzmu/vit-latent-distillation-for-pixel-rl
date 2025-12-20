from __future__ import annotations
import random
from typing import List, Optional

import dm_env
import mujoco
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs
import metaworld


from wrappers import (
    ActionDTypeWrapper, ActionRepeatWrapper, ExtendedTimeStepWrapper,
    FrameStackWrapper, NoisyMaskWrapper, SegmentationFilter,
    SegmentationToRobotMaskWrapper, SlimMaskWrapper, StackRGBAndMaskWrapper,
    get_env_action_spec, get_env_observation_spec,
)

MJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


class FlipPixelsWrapper(dm_env.Environment):


    def __init__(self, env: dm_env.Environment, keys=("pixels", "segmentation")):
        self._env = env
        self._keys = set(keys)

    # ---- required dm_env.Environment API ----
    def reset(self) -> dm_env.TimeStep:
        return self._flip_ts(self._env.reset())

    def step(self, action) -> dm_env.TimeStep:
        return self._flip_ts(self._env.step(action))

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        if hasattr(self._env, "reward_spec"):
            return self._env.reward_spec()
        return super().reward_spec()

    # ---- utilities ----
    def _flip_ts(self, ts: dm_env.TimeStep) -> dm_env.TimeStep:
        obs = ts.observation
        if isinstance(obs, dict):
            flipped = {}
            for k, v in obs.items():
                if k in self._keys and getattr(v, "ndim", 0) >= 2:
                    flipped[k] = np.flip(v, 0).copy(order="C")
                else:
                    flipped[k] = v
            obs = flipped
        elif getattr(obs, "ndim", 0) >= 2:
            obs = np.flip(obs, 0).copy(order="C")
        return ts._replace(observation=obs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class _RendererPhysics:
    def __init__(self, env):
        self._env = env
        self._cache = {}

    def _get_renderer(self, h, w):
        key = (int(h), int(w))
        r = self._cache.get(key)
        if r is None:
            r = mujoco.Renderer(self._env.model, height=int(h), width=int(w))
            self._cache[key] = r
        return r

    def _resolve_camera(self, camera_name=None):
        return camera_name if camera_name is not None else 0

    def render(self, *, height, width, camera_name=None,
               depth=False, segmentation=False, **_):
        r = self._get_renderer(height, width)
        cam = self._resolve_camera(camera_name)

        if segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(self._env.data, camera=cam)
            seg = r.render()
            if seg.ndim == 3 and seg.shape[-1] >= 2:
                seg = seg[..., [1, 0]]
            return seg

        r.disable_segmentation_rendering()
        r.update_scene(self._env.data, camera=cam)
        rgb = r.render()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb


class MT_Wrapper(dm_env.Environment):
    NUM_PROPRIO = 7

    @staticmethod
    def _first(x):
        return x[0] if isinstance(x, tuple) else x

    @staticmethod
    def _step_unpack(step_ret):
        if len(step_ret) == 5:
            obs, rew, terminated, truncated, info = step_ret
            info = dict(info) if isinstance(info, dict) else {}
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
            return obs, rew, info
        if len(step_ret) == 4:
            obs, rew, done, info = step_ret
            info = dict(info) if isinstance(info, dict) else {}
            info["done"] = bool(done)
            return obs, rew, info
        return step_ret

    @staticmethod
    def _proprio_spec(base: specs.BoundedArray, n: int) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(n,),
            dtype=base.dtype,
            minimum=base.minimum[:n],
            maximum=base.maximum[:n],
            name="observation",
        )

    def __init__(self, env_name, *, discount, seed, proprio):
        self.discount, self.proprio = discount, proprio
        key = env_name
        if key in ["MT10", "mt10"]:
            self.mt = metaworld.MT10(seed=seed)
        elif key in ["MT50", "mt50"]:
            self.mt = metaworld.MT50(seed=seed)
        elif key in ["MT3-Customized", "mt3-customized"]:
            self.mt = MT3_Customized(seed=seed)
        elif key in ["MT5-Customized", "mt5-customized"]:
            self.mt = MT5_Customized(seed=seed)
        elif key in ["MT10-Customized", "mt10-customized"]:
            self.mt = MT10_Customized(seed=seed)
        elif key in ["MT-Door", "mt-door"]:
            self.mt = MT_Door(seed=seed)
        else:
            self.mt = metaworld.MT1(key, seed=seed)

        self._train_classes = getattr(self.mt, "_train_classes", getattr(self.mt, "train_classes"))
        self._train_tasks   = getattr(self.mt, "_train_tasks",   getattr(self.mt, "train_tasks"))

        self.all_envs = {n: c() for n, c in self._train_classes.items()}

        self.robot_segmentation_ids: List[int] = list(range(8, 35))

        self._reset_flag = True
        self.cur_step = 0
        self.physics: Optional[_RendererPhysics] = None
        self._obs_spec = self._act_spec = None

    def reset(self):
        self._reset_flag, self.cur_step = False, 0
        self.env_name, self._env = random.choice(list(self.all_envs.items()))
        self._env.set_task(random.choice([t for t in self._train_tasks if t.env_name == self.env_name]))
        obs = self._first(self._env.reset())
        self.physics = _RendererPhysics(self._env)
        self.robot_segmentation_ids = list(range(8, 35))
        obs = self._maybe_slice(obs).astype(self._env.observation_space.dtype)
        return dm_env.restart(obs)

    def step(self, a):
        if self._reset_flag:
            return self.reset()
        obs, rew, info = self._step_unpack(self._env.step(a))
        self.cur_step += 1
        obs = self._maybe_slice(obs).astype(self._env.observation_space.dtype)
        ts = dm_env.transition({"reward": rew, "success": info.get("success", 0)}, obs, self.discount)
        if self.cur_step >= self._env.max_path_length:
            self._reset_flag = True
            ts = dm_env.truncation(ts.reward, obs, self.discount)
        return ts

    def observation_spec(self):
        if self._obs_spec is None:
            base = get_env_observation_spec(next(iter(self.all_envs.values())))
            if self.proprio:
                base = self._proprio_spec(base, self.NUM_PROPRIO)
            self._obs_spec = base
        return self._obs_spec

    def action_spec(self):
        if self._act_spec is None:
            self._act_spec = get_env_action_spec(next(iter(self.all_envs.values())))
        return self._act_spec

    def _maybe_slice(self, obs): return obs[: self.NUM_PROPRIO] if self.proprio else obs
    def __getattr__(self, n):    return getattr(self._env, n)


def make_metaworld(
    name: str,
    frame_stack: int = 1,
    action_repeat: int = 1,
    discount: float = 1.0,
    seed: int = 0,
    camera_name: str | None = None,
    add_segmentation_to_obs: bool = False,
    noisy_mask_drop_prob: float = 0.0,
    use_rgbm: bool = False,
    slim_mask_cfg=None,
    robot_ids: List[int] | None = None,
):
    env = MT_Wrapper(name, discount=discount, seed=seed, proprio=True)
    env.reset()

    if robot_ids is not None:
        env.robot_segmentation_ids = list(robot_ids)

    cam_kwargs = {}
    cam_kwargs["camera_name"] = camera_name if camera_name is not None else 0

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys, rgb_key = [], "pixels"
    env = pixels.Wrapper(
        env, pixels_only=False,
        render_kwargs=dict(height=84, width=84, **cam_kwargs), #was 84
        observation_key=rgb_key,
    )
    frame_keys.append(rgb_key)

    if add_segmentation_to_obs:
        seg_key = "segmentation"
        env = pixels.Wrapper(
            env, pixels_only=False,
            render_kwargs=dict(height=84*3, width=84*3, segmentation=True, **cam_kwargs), #was 84*3
            observation_key=seg_key,
        )

        env = SegmentationToRobotMaskWrapper(env, seg_key)
        env = SegmentationFilter(env, seg_key)

        if noisy_mask_drop_prob:
            env = NoisyMaskWrapper(env, seg_key, prob_drop=noisy_mask_drop_prob)
        if slim_mask_cfg and getattr(slim_mask_cfg, "use_slim_mask", False):
            env = SlimMaskWrapper(env, seg_key, slim_mask_cfg.scale, slim_mask_cfg.threshold, slim_mask_cfg.sigma)

        if use_rgbm:
            env = StackRGBAndMaskWrapper(env, rgb_key, seg_key, new_key="pixels")
            frame_keys = ["pixels"]
        else:
            frame_keys.append(seg_key)
    # print(camera_name)
    # exit()
    env = FlipPixelsWrapper(env, keys=tuple(frame_keys))

    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
