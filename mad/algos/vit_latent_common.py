from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from camera_aliases import normalize_camera_names, resolve_camera_alias_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pretrain_models import MultiViewVST, MultiViewVSMAE  # noqa: E402


def _resolve_checkpoint_path(path_str: str) -> Path:
    checkpoint = Path(path_str).expanduser()
    if checkpoint.is_file():
        return checkpoint.resolve()
    repo_candidate = REPO_ROOT / checkpoint
    if repo_candidate.is_file():
        return repo_candidate.resolve()
    raise FileNotFoundError(f"Could not find ViT checkpoint: {path_str}")


def _load_mae_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} did not contain a usable state_dict.")
    if any(key.startswith("module.") for key in payload.keys()):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return payload


def build_augmented_subsets(camera_names: Sequence[str], mode: str) -> list[list[str]]:
    cameras = list(camera_names)
    if mode == "full_and_singles":
        return [[cam] for cam in cameras]
    if mode == "all_nonempty":
        subsets: list[list[str]] = []
        for subset_len in range(1, len(cameras)):
            for subset in itertools.combinations(cameras, subset_len):
                subsets.append(list(subset))
        return subsets
    raise ValueError(f"Unknown subset mode: {mode}")


class FrozenVitLatentEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.trainable = bool(getattr(cfg, "vit_finetune_encoder", False))
        self.camera_alias_profile = resolve_camera_alias_profile(
            getattr(cfg, "agent", None),
            getattr(cfg, "camera_alias_profile", "auto"),
        )
        self.original_cameras = list(cfg.cameras)
        self.all_cameras = normalize_camera_names(self.original_cameras, self.camera_alias_profile)
        self.camera_to_slot = {name: idx for idx, name in enumerate(self.all_cameras)}
        assert len(self.camera_to_slot) == len(self.all_cameras), "Camera names must be unique."
        self.active_cameras = list(self.all_cameras)

        num_views, channels, height, width = cfg.obs_shape
        encoder = MultiViewVST(
            image_size=(height, width),
            patch_size=cfg.vit_patch_size,
            dim=cfg.vit_dim,
            depth=cfg.vit_depth,
            heads=cfg.vit_heads,
            mlp_dim=cfg.vit_mlp_dim,
            channels=channels,
            num_views=num_views,
            frame_stack=cfg.frame_stack,
        )
        self.mae = MultiViewVSMAE(
            encoder=encoder,
            decoder_dim=cfg.vit_decoder_dim,
            decoder_depth=cfg.vit_decoder_depth,
            decoder_heads=cfg.vit_decoder_heads,
            use_sincosmod_encodings=True,
            frame_stack=cfg.frame_stack,
            num_views=num_views,
        ).to(self.device)

        checkpoint_path = _resolve_checkpoint_path(cfg.vit_checkpoint)
        state_dict = _load_mae_state_dict(checkpoint_path, self.device)
        self.mae.load_state_dict(state_dict, strict=True)
        if self.trainable:
            self.mae.train()
        else:
            self.mae.eval()
            for param in self.mae.parameters():
                param.requires_grad = False

        self.repr_dim = int(self.mae.encoder_dim)
        self.num_views = int(num_views)
        self.image_channels = int(channels)
        self.image_height = int(height)
        self.image_width = int(width)
        self.checkpoint_path = checkpoint_path

    def train(self, mode: bool = True):
        # Frozen ViTs stay in eval mode; finetuned ViTs follow the caller's train/eval mode.
        super().train(mode if self.trainable else False)
        if self.trainable:
            self.mae.train(mode)
        else:
            self.mae.eval()
        return self

    def set_active_cameras(self, camera_names: Sequence[str]) -> None:
        names = normalize_camera_names(list(camera_names), self.camera_alias_profile)
        assert names, "Active camera list must be non-empty."
        missing = [name for name in names if name not in self.camera_to_slot]
        assert not missing, f"Unknown camera names {missing}; known={self.all_cameras}"
        self.active_cameras = names

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=torch.float32)
        if obs.max().item() > 1.5:
            obs = obs / 255.0
        return obs

    def _resolve_camera_names(self, num_observed_views: int, camera_names: Sequence[str] | None) -> list[str]:
        if camera_names is not None:
            names = normalize_camera_names(list(camera_names), self.camera_alias_profile)
        elif num_observed_views == len(self.active_cameras):
            names = list(self.active_cameras)
        elif num_observed_views == len(self.all_cameras):
            names = list(self.all_cameras)
        else:
            raise ValueError(
                f"Could not infer camera names for obs with {num_observed_views} views. "
                f"active_cameras={self.active_cameras} all_cameras={self.all_cameras}"
            )
        assert len(names) == num_observed_views, (
            f"Camera name count {len(names)} does not match observation view count {num_observed_views}"
        )
        return names

    def _scatter_obs(
        self, obs: torch.Tensor, camera_names: Sequence[str] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert obs.ndim == 5, f"Expected obs [B,V,C,H,W], got {tuple(obs.shape)}"
        batch_size, observed_views, channels, height, width = obs.shape
        assert channels == self.image_channels, f"Expected C={self.image_channels}, got {channels}"
        assert (height, width) == (self.image_height, self.image_width), (
            f"Expected {(self.image_height, self.image_width)}, got {(height, width)}"
        )
        names = self._resolve_camera_names(observed_views, camera_names)
        full_obs = torch.zeros(
            (batch_size, self.num_views, channels, height, width),
            device=obs.device,
            dtype=obs.dtype,
        )
        visible = torch.zeros((batch_size, self.num_views), device=obs.device, dtype=torch.bool)
        for obs_idx, name in enumerate(names):
            slot = self.camera_to_slot[name]
            full_obs[:, slot] = obs[:, obs_idx]
            visible[:, slot] = True
        return full_obs, visible

    def encode_observation(
        self, obs: torch.Tensor, camera_names: Sequence[str] | None = None, grad: bool = False
    ) -> torch.Tensor:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            obs = self._normalize_obs(obs)
            full_obs, visible = self._scatter_obs(obs, camera_names=camera_names)
            return self.mae.get_embeddings(full_obs, visible_views=visible, eval=not grad)

    def encode_named_subset(
        self, full_obs: torch.Tensor, subset_camera_names: Sequence[str], grad: bool = False
    ) -> torch.Tensor:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            obs = self._normalize_obs(full_obs)
            full_obs_scattered, _ = self._scatter_obs(obs, camera_names=self.all_cameras)
            visible = torch.zeros((obs.shape[0], self.num_views), device=obs.device, dtype=torch.bool)
            for name in subset_camera_names:
                visible[:, self.camera_to_slot[name]] = True
            return self.mae.get_embeddings(full_obs_scattered, visible_views=visible, eval=not grad)


class Actor(nn.Module):
    def __init__(self, state_shape, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim + state_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )
        self.log_std_bounds = log_std_bounds
        self.apply(utils.weight_init)

    def forward(self, state, obs):
        h = self.trunk(obs)
        h = torch.cat([state, h], dim=1)
        mu, log_std = self.policy(h).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        return utils.SquashedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_shape, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0] + state_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0] + state_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(utils.weight_init)

    def forward(self, state, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action, state], dim=-1)
        return self.Q1(h_action), self.Q2(h_action)


class FrozenVitSacBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.encoder_trainable = bool(getattr(cfg, "vit_finetune_encoder", False))
        self.vit_unfreeze_after_steps = int(getattr(cfg, "vit_unfreeze_after_steps", 0))
        self.vit_anchor_reg_weight = float(getattr(cfg, "vit_anchor_reg_weight", 0.0))
        self.vit_subset_alignment_reg_weight = float(getattr(cfg, "vit_subset_alignment_reg_weight", 0.0))
        self.critic_target_tau = cfg.critic_target_tau
        self.update_every_steps = cfg.update_every_steps
        self.num_expl_steps = cfg.num_expl_steps

        self.encoder = FrozenVitLatentEncoder(cfg).to(cfg.device)
        self.anchor_encoder = None
        if self.encoder_trainable and self.vit_anchor_reg_weight > 0.0:
            self.anchor_encoder = copy.deepcopy(self.encoder).to(cfg.device)
            self.anchor_encoder.trainable = False
            self.anchor_encoder.eval()
            for param in self.anchor_encoder.parameters():
                param.requires_grad = False
        self.actor = Actor(
            cfg.state_shape,
            self.encoder.repr_dim,
            cfg.action_shape,
            cfg.feature_dim,
            cfg.hidden_dim,
            cfg.log_std_bounds,
        ).to(cfg.device)
        self.critic = Critic(
            cfg.state_shape,
            self.encoder.repr_dim,
            cfg.action_shape,
            cfg.feature_dim,
            cfg.hidden_dim,
        ).to(cfg.device)
        self.critic_target = Critic(
            cfg.state_shape,
            self.encoder.repr_dim,
            cfg.action_shape,
            cfg.feature_dim,
            cfg.hidden_dim,
        ).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature), device=cfg.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -cfg.action_shape[0]

        self.encoder_opt = None
        if self.encoder_trainable:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            assert encoder_params, "Encoder finetuning requested, but no trainable ViT parameters were found."
            self.encoder_opt = torch.optim.Adam(encoder_params, lr=cfg.vit_encoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training and self.encoder_trainable)
        if self.anchor_encoder is not None:
            self.anchor_encoder.train(False)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _ensure_finite(self, name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite values detected in {name}.")

    def _module_grad_norm(self, module: nn.Module) -> float:
        sq_sum = 0.0
        saw_grad = False
        for param in module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            sq_sum += float(torch.sum(grad * grad).item())
            saw_grad = True
        return sq_sum ** 0.5 if saw_grad else 0.0

    def _module_param_norm(self, module: nn.Module) -> float:
        sq_sum = 0.0
        saw_param = False
        for param in module.parameters():
            data = param.detach()
            sq_sum += float(torch.sum(data * data).item())
            saw_param = True
        return sq_sum ** 0.5 if saw_param else 0.0

    def _record_latent_metrics(self, metrics: dict, prefix: str, tensor: torch.Tensor) -> None:
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        metrics[f"{prefix}_norm"] = float(flat.norm(dim=1).mean().item())
        metrics[f"{prefix}_std"] = float(flat.std(unbiased=False).item())
        metrics[f"{prefix}_finite_frac"] = float(torch.isfinite(flat).float().mean().item())

    def set_active_cameras(self, camera_names: Sequence[str]) -> None:
        self.encoder.set_active_cameras(camera_names)

    def _encoder_update_active(self, step: int) -> bool:
        return self.encoder_trainable and step >= self.vit_unfreeze_after_steps

    def _set_encoder_update_mode(self, update_active: bool) -> None:
        if self.encoder_trainable:
            self.encoder.train(self.training and update_active)
        else:
            self.encoder.train(False)

    def _compute_anchor_reg(
        self, obs: torch.Tensor, subset_camera_names: Sequence[str], current_latent: torch.Tensor
    ) -> torch.Tensor:
        if self.anchor_encoder is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            anchor_latent = self.anchor_encoder.encode_named_subset(obs, subset_camera_names, grad=False)
        anchor_latent = anchor_latent.detach()
        self._ensure_finite("anchor_latent", anchor_latent)
        return F.mse_loss(current_latent, anchor_latent)

    def act(self, state, obs, step, eval_mode):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        with utils.eval_mode(self.encoder):
            with torch.inference_mode():
                z = self.encoder.encode_observation(obs)
        dist = self.actor(state, z)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def _standard_update_critic(
        self,
        state,
        z,
        action,
        reward,
        discount,
        next_state,
        next_z,
        encoder_reg_loss: torch.Tensor | None = None,
        update_encoder: bool = False,
    ):
        metrics = {}
        with torch.no_grad():
            dist = self.actor(next_state, next_z)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_state, next_z, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)
            self._ensure_finite("target_Q", target_Q)

        Q1, Q2 = self.critic(state, z, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self._ensure_finite("critic_loss", critic_loss)
        if encoder_reg_loss is None:
            encoder_reg_loss = torch.tensor(0.0, device=self.device)
        self._ensure_finite("encoder_reg_loss", encoder_reg_loss)
        total_loss = critic_loss + encoder_reg_loss
        self._ensure_finite("critic_total_loss", total_loss)

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        metrics["critic_grad_norm"] = self._module_grad_norm(self.critic)
        metrics["encoder_param_norm"] = self._module_param_norm(self.encoder)
        if self.encoder_opt is not None:
            metrics["encoder_grad_norm"] = self._module_grad_norm(self.encoder)
        if update_encoder and self.encoder_opt is not None:
            self.encoder_opt.step()
        self.critic_opt.step()

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["critic_total_loss"] = total_loss.item()
        return metrics

    def _standard_update_actor(self, state, z):
        metrics = {}
        dist = self.actor(state, z)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(state, z, action)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        self._ensure_finite("actor_loss", actor_loss)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        metrics["actor_grad_norm"] = self._module_grad_norm(self.actor)
        self.actor_opt.step()

        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["actor_entropy"] = -log_prob.mean().item()
        metrics["actor_target_entropy"] = self.target_entropy
        metrics["alpha_loss"] = alpha_loss.item()
        metrics["alpha_value"] = self.alpha.item()
        return metrics
