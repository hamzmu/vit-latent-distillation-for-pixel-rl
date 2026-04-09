# Meta-World v2 Wrapper

from collections import deque
from typing import Any, NamedTuple
from pathlib import Path
import sys
import mujoco
import numpy as np

import dm_env
from dm_env import specs, TimeStep, StepType
try:
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    GOAL_OBSERVABLE_ENVS = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    GOAL_OBSERVABLE_SUFFIX = "-v2-goal-observable"
except ImportError:
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    GOAL_OBSERVABLE_ENVS = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    GOAL_OBSERVABLE_SUFFIX = "-v3-goal-observable"

from camera_aliases import normalize_camera_names, normalize_task_name, resolve_camera_alias_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from metaworld_dm_env import make_metaworld as make_our_metaworld


def make(cfg):
    backend = getattr(cfg, "metaworld_backend", "ours")
    if backend == "ours":
        return make_with_our_backend(cfg)
    if backend != "mad":
        raise ValueError(f"Unknown metaworld backend: {backend}")

    task = normalize_task_name(cfg.task)
    frame_stack = cfg.frame_stack
    seed = cfg.seed
    cameras = cfg.cameras
    img_size = cfg.img_size
    camera_profile = resolve_camera_alias_profile(
        getattr(cfg, "agent", None),
        getattr(cfg, "camera_alias_profile", "auto"),
    )
    

    action_repeat = 2
    max_episode_steps = 200
    sparse_rewards = getattr(cfg, 'sparse_rewards', False)

    task += GOAL_OBSERVABLE_SUFFIX

    if task not in GOAL_OBSERVABLE_ENVS:
        raise ValueError(f'Invalid MetaWorld task: {task}')

    env = GOAL_OBSERVABLE_ENVS[task](seed=seed)
    env._freeze_rand_vec = False
    env = Gym2DMC(env, max_episode_steps = max_episode_steps, sparse_rewards=sparse_rewards) 
    env = ActionWrapper(env, action_repeat=action_repeat, dtype=np.float32, minimum=-1.0, maximum=1.0)
    env = PixelWrapper(env, img_size, cameras=cameras, camera_alias_profile=camera_profile)
    env = FrameStackWrapper(env, frame_stack)   
    env = ExtendedTimeStepWrapper(env)

    env.reset()
    return env


def _to_our_task_name(task: str) -> str:
    normalized = normalize_task_name(task)
    if normalized in ("MT10", "mt10", "MT50", "mt50"):
        return normalized
    if normalized.endswith("-v3"):
        return normalized
    return f"{normalized}-v3"


class OurMetaWorldAdapter(dm_env.Environment):
    def __init__(self, cfg, camera_alias_profile: str):
        self._cfg = cfg
        self._camera_alias_profile = camera_alias_profile
        self._seed = cfg.seed
        self._frame_stack = int(cfg.frame_stack)
        self._action_repeat = 2
        self._task_name = _to_our_task_name(cfg.task)
        self._camera_names = list(cfg.cameras)
        self._is_success = 0
        self._last_pixels = None
        self._frames = deque([], maxlen=self._frame_stack)
        self._state_spec = None
        self._observation_spec = None
        self._build_env()

    def _build_env(self):
        self._resolved_cameras = normalize_camera_names(self._camera_names, self._camera_alias_profile)
        self._env = make_our_metaworld(
            name=self._task_name,
            frame_stack=self._frame_stack,
            action_repeat=self._action_repeat,
            discount=1.0,
            seed=self._seed,
            proprio=False,
            camera_names=self._resolved_cameras,
        )
        self._pixel_keys = [self._pixel_obs_key(i) for i in range(len(self._resolved_cameras))]
        obs_spec = self._env.observation_spec()
        example = obs_spec[self._pixel_keys[0]]
        self._observation_spec = specs.BoundedArray(
            shape=(len(self._resolved_cameras), *example.shape),
            dtype=example.dtype,
            minimum=0,
            maximum=255,
            name="observation",
        )
        example_state = self._extract_accessible_state(self._env.reset().observation)
        self._state_spec = specs.Array(
            shape=(example_state.shape[0] * self._frame_stack,),
            dtype=example_state.dtype,
            name="state",
        )
        self._frames = deque([], maxlen=self._frame_stack)

    def _pixel_obs_key(self, view_idx: int) -> str:
        if view_idx == 0:
            return "pixels"
        if view_idx == 1:
            return "pixels_aux"
        return f"pixels_aux{view_idx}"

    def _stack_pixels(self, obs: dict) -> np.ndarray:
        return np.stack([obs[key] for key in self._pixel_keys], axis=0)

    def _extract_accessible_state(self, obs: dict) -> np.ndarray:
        raw_state = obs.get("observation", obs.get("state"))
        if raw_state is None:
            raise KeyError(f"Expected observation dict to contain 'observation' or 'state', got keys={list(obs.keys())}")
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape[0] >= 22:
            return np.concatenate((state[:4], state[18:22]), axis=0).astype(np.float32)
        return state.astype(np.float32)

    def _stacked_state(self) -> np.ndarray:
        assert len(self._frames) == self._frame_stack
        return np.concatenate(list(self._frames), axis=0)

    def _render_multiview_from_last_pixels(self) -> np.ndarray:
        assert self._last_pixels is not None
        tiles = []
        for vidx in range(self._last_pixels.shape[0]):
            view = self._last_pixels[vidx]
            rgb = view[:3].transpose(1, 2, 0)
            tiles.append(rgb)
        return np.concatenate(tiles, axis=1)

    @property
    def stacked_state(self):
        return self._stacked_state()

    @property
    def is_success(self):
        return int(self._is_success)

    def change_cams(self, new_cams):
        self._camera_names = list(new_cams)
        try:
            self._env.close()
        except Exception:
            pass
        self._build_env()

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation
        state = self._extract_accessible_state(obs)
        self._frames.clear()
        for _ in range(self._frame_stack):
            self._frames.append(state)
        self._last_pixels = self._stack_pixels(obs)
        self._is_success = int(time_step.reward["success"]) if isinstance(time_step.reward, dict) else 0
        return ExtendedTimeStep(
            observation=self._last_pixels,
            step_type=time_step.step_type,
            reward=float(time_step.reward["reward"]) if isinstance(time_step.reward, dict) else float(time_step.reward or 0.0),
            discount=float(time_step.discount or 1.0),
            action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
            state=self._stacked_state(),
        )

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation
        state = self._extract_accessible_state(obs)
        self._frames.append(state)
        self._last_pixels = self._stack_pixels(obs)
        reward = time_step.reward
        if isinstance(reward, dict):
            self._is_success = int(reward.get("success", 0))
            reward_val = float(reward.get("reward", 0.0))
        else:
            reward_val = float(reward or 0.0)
        return ExtendedTimeStep(
            observation=self._last_pixels,
            step_type=time_step.step_type,
            reward=reward_val,
            discount=float(time_step.discount or 1.0),
            action=action.astype(self.action_spec().dtype, copy=False),
            state=self._stacked_state(),
        )

    def render_multiview(self):
        return self._render_multiview_from_last_pixels()

    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_with_our_backend(cfg):
    camera_profile = resolve_camera_alias_profile(
        getattr(cfg, "agent", None),
        getattr(cfg, "camera_alias_profile", "auto"),
    )
    return OurMetaWorldAdapter(cfg, camera_alias_profile=camera_profile)


class Gym2DMC(dm_env.Environment):
    """
        Convert a Gym environment to a DMC environment, adds a state and success info, 
        limits episode to max_episode_steps, chooses sparse or dense rewards
    """
    def __init__(self, gym_env, max_episode_steps, sparse_rewards=False) -> None:
        gym_obs_space = gym_env.observation_space
        self._observation_spec = specs.BoundedArray(
            shape=gym_obs_space.shape,
            dtype=gym_obs_space.dtype,
            minimum=gym_obs_space.low,
            maximum=gym_obs_space.high,
            name='observation'
            )
        gym_act_space = gym_env.action_space
        self._action_spec = specs.BoundedArray(
            shape=gym_act_space.shape,
            dtype=gym_act_space.dtype,
            minimum=gym_act_space.low,
            maximum=gym_act_space.high,
            name='action'
            )
        self._env = gym_env
        # Success info
        self._is_success = 0
        # State info
        self._state_obs = None
        self.reset()
        self._state_spec = specs.Array(
            shape=self.state.shape,
            dtype=self.state.dtype,
            name='state'
        )
        # Reward Type
        self._sparse_rewards = sparse_rewards
        # Time Limit
        self._elapsed_steps = None
        self._max_episode_steps = max_episode_steps



    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if self._sparse_rewards:
            reward = info['success']
        self._state_obs = obs.astype(np.float32)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            step_type = StepType.LAST
            discount = 0.0
            self._is_success = info['success']
        else:
            step_type = StepType.MID
            discount = 1.0
        return TimeStep(step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=obs)

    def reset(self):
        obs = self._env.reset()
        obs = self._env.step(np.zeros_like(self._env.action_space.sample()))[0]
        self._state_obs = obs.astype(np.float32)
        self._is_success = 0  # Adding success metric
        self._elapsed_steps = 0
        return TimeStep(step_type=StepType.FIRST,
                        reward=0.0,
                        discount=1.0,
                        observation=obs)

    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        accessible_state = np.concatenate((state[:4], state[18 : 18 + 4])) # Choose only robot-accessible state
        # accessible_state = state # Full state
        return accessible_state

    @property
    def is_success(self):
        return self._is_success

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._action_spec
    
    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionWrapper(dm_env.Environment):
    """ Action repeat, clip, and cast"""
    def __init__(self, env, action_repeat, dtype, minimum, maximum):
        self._env = env
        self._action_repeat = action_repeat
        self._dtype = dtype
        self._min = minimum
        self._max = maximum
        self._action_spec = self._env.action_spec().replace(dtype=self._dtype)

    def step(self, action):
        action = np.clip(action, self._min, self._max).astype(self._dtype)
        reward = 0.0
        discount = 1.0
        for i in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def state_spec(self):
        return self._env.state_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class PixelWrapper(dm_env.Environment):
    """ Pixel wrapper """
    def __init__(self, env, img_size, cameras=['corner2'], camera_alias_profile='mad'):
        super().__init__()
        self._env = env
        self._cameras = cameras
        self._img_size = img_size
        self._camera_alias_profile = camera_alias_profile

        self._observation_spec = specs.BoundedArray(
                        shape=(len(self._cameras), 3, img_size, img_size),
                        dtype=np.uint8, 
                        minimum=0, 
                        maximum=255,
                        name="observation")
        

        # Updating camera logic
        self.change_cams(self._cameras)
        # Third1 Person Corner camera
        self._env.unwrapped.model.cam_pos[self._camera_id("corner2")] = [0.75, 0.075, 0.7]
        # Third2 (Top-dow)n view camera
        self._env.unwrapped.model.cam_pos[self._camera_id("corner3")] = [0.72, 0.1, 1.2]
        # Front Behind Grippper Camera
        self._env.unwrapped.model.cam_fovy[self._camera_id("behindGripper")] = 90


    def reset(self):
        time_step = self._env.reset()
        return time_step._replace(observation=self._get_pixel_obs())

    def step(self, action):
        time_step = self._env.step(action)
        return time_step._replace(observation=self._get_pixel_obs())

 
    def _get_pixel_obs(self):
        self.update_trackcam_pos() 
        all_cams = []
        for each_cam in self._cameras:
            all_cams.append(
                self.render(width=self._img_size, height=self._img_size, camera_name=each_cam).transpose(2, 0, 1) # C, H, W
            )
        return np.stack(all_cams, axis=0) # V, C, H, W

    # Track cams in Mujoco need to be updated at every step, while fixed cams can be updated only once
    def _camera_id(self, camera_name: str) -> int:
        return mujoco.mj_name2id(self._env.unwrapped.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    def update_trackcam_pos(self):
        # shift camera pos
        self._env.unwrapped.data.cam_xpos[self._camera_id("gripperPOV")] += [0.045, -0.025, -0.04]
        # shift camera pos
        self._env.unwrapped.data.cam_xpos[self._camera_id("behindGripper")] += [0.0, 0.2, 0.00]
        # rotate camera 180
        yaw = np.radians(180)
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        cam_xmat = np.reshape(self._env.unwrapped.data.cam_xmat[self._camera_id("gripperPOV")], (3, 3))
        self._env.unwrapped.data.cam_xmat[self._camera_id("gripperPOV")] = np.dot(cam_xmat, rotation_matrix).flatten()

        cam_xmat = np.reshape(self._env.unwrapped.data.cam_xmat[self._camera_id("behindGripper")], (3, 3))
        self._env.unwrapped.data.cam_xmat[self._camera_id("behindGripper")] = np.dot(cam_xmat, rotation_matrix).flatten()


    # Utility, updating camera logic
    def change_cams(self, new_cams):
        self._cameras = normalize_camera_names(list(new_cams), self._camera_alias_profile)

    # Renders with all cameras for visualization
    def render_multiview(self):
        all_cams = []
        for each_cam in self._cameras:
            all_cams.append(
                self.render(camera_name=each_cam)# H, W, C
            )
        return np.concatenate(all_cams, axis=1) # Extended view

    def render(self, width=256, height=256, camera_name='corner2'):
        try:
            return self._env.unwrapped.render(
                offscreen=True,
                resolution=(int(width), int(height)),
                camera_name=camera_name,
            ).copy()
        except TypeError:
            renderer = self._env.unwrapped.mujoco_renderer
            renderer.width = int(width)
            renderer.height = int(height)
            renderer.camera_id = self._camera_id(camera_name)
            return renderer.render(render_mode='rgb_array').copy()

    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._env.state_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    


class FrameStackWrapper(dm_env.Environment):
    """ Framestacks pixel and state wrapper """
    def __init__(self, env, frame_stack):
        super().__init__()
        self._env = env
        self._frame_stack = frame_stack


        self._frames = deque([], maxlen=self._frame_stack)
        obs_spec = self._env.observation_spec()
        self._observation_spec = specs.BoundedArray(
                        shape=(obs_spec.shape[0], self._frame_stack*obs_spec.shape[1], *obs_spec.shape[2:]),
                        dtype=obs_spec.dtype, 
                        minimum=obs_spec.minimum, 
                        maximum=obs_spec.maximum,
                        name=obs_spec.name)
        
        self._states = deque([], maxlen=self._frame_stack)
        state_spec = self._env.state_spec()
        self._state_spec = specs.Array(
                        shape=(self._frame_stack * state_spec.shape[0],),
                        dtype=state_spec.dtype, 
                        name=state_spec.name)


    def reset(self):
        time_step = self._env.reset()
        for _ in range(self._frame_stack):
            self._frames.append(time_step.observation)
            self._states.append(self._env.state)
        return time_step._replace(observation=self._stacked_obs()) 

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(time_step.observation)
        self._states.append(self._env.state)
        return time_step._replace(observation=self._stacked_obs())


    @property
    def stacked_state(self):
        return self._stacked_state()

    def _stacked_obs(self):
        assert len(self._frames) == self._frame_stack
        return np.concatenate(list(self._frames), axis=1) 

    def _stacked_state(self):
        assert len(self._states) == self._frame_stack
        return np.concatenate(list(self._states), axis=0) 
 
    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    



class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    state: Any

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
        


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

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
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                action=action,
                                state=self._env.stacked_state)

    def observation_spec(self):
        return self._env.observation_spec()

    def state_spec(self):
        return self._env.state_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
