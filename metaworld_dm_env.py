# =========================
# metaworld_dm_env.py
# (NO segmentation anywhere)
# Provides obs keys: "pixels" and "pixels_aux"
# =========================
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
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
    get_env_action_spec,
    get_env_observation_spec,
)


class FlipPixelsWrapper(dm_env.Environment):
    def __init__(self, env: dm_env.Environment, keys=("pixels", "pixels_aux")):
        self._env = env
        self._keys = set(keys)

    def reset(self) -> dm_env.TimeStep:
        return self._flip_ts(self._env.reset())

    def step(self, action) -> dm_env.TimeStep:
        return self._flip_ts(self._env.step(action))

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

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
    """
    Physics-like object with .render(...) compatible with dm_control pixels.Wrapper.
    Uses mujoco.Renderer so we can render multiple cameras by name.
    """
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

    def render(self, *, height, width, camera_name=None, **_):
        r = self._get_renderer(height, width)
        cam = self._resolve_camera(camera_name)
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
        else:
            self.mt = metaworld.MT1(key, seed=seed)

        self._train_classes = getattr(self.mt, "_train_classes", getattr(self.mt, "train_classes"))
        self._train_tasks = getattr(self.mt, "_train_tasks", getattr(self.mt, "train_tasks"))

        self.all_envs = {n: c() for n, c in self._train_classes.items()}
        self._reset_flag = True
        self.cur_step = 0
        self.physics: Optional[_RendererPhysics] = None
        self._obs_spec = self._act_spec = None

    def reset(self):
        self._reset_flag, self.cur_step = False, 0
        self.env_name, self._env = random.choice(list(self.all_envs.items()))
        self._env.set_task(random.choice([t for t in self._train_tasks if t.env_name == self.env_name]))
        obs = self._first(self._env.reset())

        # Provide .render(...) used by pixels.Wrapper
        self.physics = _RendererPhysics(self._env)

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

    def _maybe_slice(self, obs):
        return obs[: self.NUM_PROPRIO] if self.proprio else obs

    def __getattr__(self, n):
        return getattr(self._env, n)


def make_metaworld(
    name: str,
    frame_stack: int = 1,
    action_repeat: int = 1,
    discount: float = 1.0,
    seed: int = 0,
    camera_name: str | None = None,
    camera_aux_name: str | None = None,
    add_aux_pixels_to_obs: bool = True,
):
    env = MT_Wrapper(name, discount=discount, seed=seed, proprio=True)
    env.reset()

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys: List[str] = []

    # main camera
    rgb_key = "pixels"
    cam_kwargs_main = {"camera_name": camera_name if camera_name is not None else 0}
    env = pixels.Wrapper(
        env,
        pixels_only=False,
        render_kwargs=dict(height=84, width=84, **cam_kwargs_main),
        observation_key=rgb_key,
    )
    frame_keys.append(rgb_key)

    # aux camera
    if add_aux_pixels_to_obs:
        aux_key = "pixels_aux"
        cam_kwargs_aux = {"camera_name": camera_aux_name if camera_aux_name is not None else 0}
        env = pixels.Wrapper(
            env,
            pixels_only=False,
            render_kwargs=dict(height=84, width=84, **cam_kwargs_aux),
            observation_key=aux_key,
        )
        frame_keys.append(aux_key)

    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
