import random
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control.mujoco import engine
from dm_env import StepType, specs

import gymnasium as gym


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats, use_metaworld_reward_dict=False):
        self._env = env
        self._num_repeats = num_repeats
        self.use_metaworld_reward_dict = use_metaworld_reward_dict

    def step(self, action):
        if self.use_metaworld_reward_dict:
            reward = 0.0
            success = False
            discount = 1.0
            for _ in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward["reward"] or 0.0) * discount
                success = success or time_step.reward["success"]
                discount *= time_step.discount
                if time_step.last():
                    break
            reward_dict = {"reward": reward, "success": success}
            return time_step._replace(reward=reward_dict, discount=discount)
        else:
            reward = 0.0
            discount = 1.0
            for _ in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward or 0.0) * discount
                discount *= time_step.discount
                if time_step.last():
                    break
            return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, frame_keys=None):
        self._env = env
        self._num_frames = num_frames

        if frame_keys is None:
            frame_keys = ["pixels"]
        if not isinstance(frame_keys, list):
            frame_keys = [frame_keys]
        self._frame_keys = frame_keys

        self._frames = [deque([], maxlen=num_frames) for _ in range(len(frame_keys))]

        wrapped_obs_spec = env.observation_spec()
        for key in frame_keys:
            assert key in wrapped_obs_spec

            frame_shape = wrapped_obs_spec[key].shape
            frame_dtype = wrapped_obs_spec[key].dtype
            # remove batch dim
            if len(frame_shape) == 4:
                frame_shape = frame_shape[1:]
            wrapped_obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate([[frame_shape[2] * num_frames], frame_shape[:2]], axis=0),
                dtype=frame_dtype,
                minimum=0,
                maximum=255,
                name="observation",
            )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        for i, key in enumerate(self._frame_keys):
            assert len(self._frames[i]) == self._num_frames
            stacked_frames = np.concatenate(list(self._frames[i]), axis=0)
            obs[key] = stacked_frames
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step, key):
        pixels = time_step.observation[key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        for i, key in enumerate(self._frame_keys):
            pixels = self._extract_pixels(time_step, key)
            for _ in range(self._num_frames):
                self._frames[i].append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        for i, key in enumerate(self._frame_keys):
            pixels = self._extract_pixels(time_step, key)
            self._frames[i].append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env, has_success_metric=False):
        self._env = env
        self.has_success_metric = has_success_metric

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        if time_step.reward is None and self.has_success_metric:
            reward = {"reward": 0.0, "success": 0}
        elif time_step.reward is None:
            reward = 0.0
        else:
            reward = time_step.reward

        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class Render_Wrapper:
    def __init__(self, render_fn):
        self._fn = render_fn

    def render(self, *args, **kwargs):
        try:
            return self._fn(*args, **kwargs)
        except TypeError:
            return self._fn()


class Camera_Render_Wrapper:
    def __init__(self, env_sim, lookat, distance, azimuth, elevation):
        self.lookat = lookat
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.env_sim = env_sim
        self.camera = None

    def render(self, height, width, *args, **kwargs):
        if self.camera is None or self.camera.height != height or self.camera.width != width:
            self.camera = engine.MovableCamera(self.env_sim, height, width)
            self.camera.set_pose(self.lookat, self.distance, self.azimuth, self.elevation)
        return self.camera.render(*args, **kwargs).copy()


class RandomCameraWrapper(dm_env.Environment):
    def __init__(self, env, lookats, distances, azimuths, elevations):
        self._env = env
        self.render_wrappers = [
            Camera_Render_Wrapper(self._env.sim, lookat, distance, azimuth, elevation)
            for lookat, distance, azimuth, elevation in zip(lookats, distances, azimuths, elevations)
        ]
        self.physics = self.sample_camera()

    def sample_camera(self):
        return random.choice(self.render_wrappers)

    def reset(self):
        self.physics = self.sample_camera()
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class CropWrapper(dm_env.Environment):
    def __init__(self, env, keys_to_crop, top_left, bottom_right):
        self._env = env
        self.keys_to_crop = keys_to_crop
        self.top_left = top_left
        self.bottom_right = bottom_right

        wrapped_obs_spec = env.observation_spec()
        for key in keys_to_crop:
            old_spec = wrapped_obs_spec[key]
            frame_shape = list(old_spec.shape)
            frame_shape[0] = bottom_right[0] - top_left[0]
            frame_shape[1] = bottom_right[1] - top_left[1]
            new_spec = specs.BoundedArray(
                shape=tuple(frame_shape),
                dtype=old_spec.dtype,
                minimum=0,
                maximum=255,
                name=old_spec.name,
            )
            wrapped_obs_spec[key] = new_spec

        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        for key in self.keys_to_crop:
            frame = obs[key][self.top_left[0] : self.bottom_right[0], self.top_left[1] : self.bottom_right[1]]
            obs[key] = frame
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def get_env_observation_spec(env):
    space = env.observation_space
    if isinstance(space, gym.spaces.Dict):
        space = space["observation"]
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name="observation",
    )


def get_env_action_spec(env):
    return specs.BoundedArray(
        shape=env.action_space.shape,
        dtype=env.action_space.dtype,
        minimum=env.action_space.low,
        maximum=env.action_space.high,
        name="action",
    )
