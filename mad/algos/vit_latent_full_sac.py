import torch

import utils
from algos.vit_latent_common import FrozenVitSacBase


class AGENT(FrozenVitSacBase):
    """
    SAC baseline on top of a frozen pretrained ViT latent.

    Training uses the full camera set available from the environment.
    Evaluation can still pass any subset through the frozen encoder via set_active_cameras().
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder.set_active_cameras(cfg.cameras)

    def update(self, replay_iter, step):
        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        state, obs, action, reward, discount, next_state, next_obs = utils.to_torch(batch, self.device)
        encoder_update_active = self._encoder_update_active(step)
        self._set_encoder_update_mode(encoder_update_active)

        if encoder_update_active:
            z = self.encoder.encode_named_subset(obs, self.encoder.all_cameras, grad=True)
            next_z = self.encoder.encode_named_subset(next_obs, self.encoder.all_cameras, grad=False)
        else:
            with utils.eval_mode(self.encoder):
                z = self.encoder.encode_named_subset(obs, self.encoder.all_cameras, grad=False)
                next_z = self.encoder.encode_named_subset(next_obs, self.encoder.all_cameras, grad=False)

        anchor_reg_loss = torch.tensor(0.0, device=self.device)
        total_reg_loss = torch.tensor(0.0, device=self.device)
        if encoder_update_active and self.vit_anchor_reg_weight > 0.0:
            anchor_reg_loss = self._compute_anchor_reg(obs, self.encoder.all_cameras, z)
            total_reg_loss = self.vit_anchor_reg_weight * anchor_reg_loss

        self._ensure_finite("latent_z", z)
        self._ensure_finite("latent_next_z", next_z)
        metrics["batch_reward"] = reward.mean().item()
        metrics["encoder_trainable"] = float(self.encoder_trainable)
        metrics["encoder_update_active"] = float(encoder_update_active)
        metrics["encoder_anchor_reg_loss"] = float(anchor_reg_loss.item())
        metrics["encoder_total_reg_loss"] = float(total_reg_loss.item())
        self._record_latent_metrics(metrics, "latent_z", z)
        self._record_latent_metrics(metrics, "latent_next_z", next_z)
        metrics.update(
            self._standard_update_critic(
                state,
                z,
                action,
                reward,
                discount,
                next_state,
                next_z,
                encoder_reg_loss=total_reg_loss,
                update_encoder=encoder_update_active,
            )
        )
        metrics.update(self._standard_update_actor(state.detach(), z.detach()))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
