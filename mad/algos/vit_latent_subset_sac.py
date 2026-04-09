import torch
import torch.nn.functional as F

import utils
from algos.vit_latent_common import FrozenVitSacBase, build_augmented_subsets


class AGENT(FrozenVitSacBase):
    """
    SAC on a frozen pretrained ViT latent with MAD-style policy robustness updates.

    - The frozen ViT provides the scene latent.
    - The critic is trained on the full latent plus subset latents against the same target.
    - The actor is trained on full + subset latents, but actions are scored against the full latent,
      matching the stabilizing idea used in MAD.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder.set_active_cameras(cfg.cameras)
        # Build subsets in the encoder's normalized camera space so alias names
        # like "first" map to the slots expected by the pretrained ViT.
        self.augmented_subsets = build_augmented_subsets(self.encoder.all_cameras, cfg.vit_subset_mode)
        self.latent_aug_alpha = float(cfg.latent_aug_alpha)
        self.latent_aug_beta = 1.0 - self.latent_aug_alpha
        self.num_repetitions = 1 + len(self.augmented_subsets)

    def _encode_training_latents(self, obs: torch.Tensor, grad: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z_full = self.encoder.encode_named_subset(obs, self.encoder.all_cameras, grad=grad)
        z_subsets = [self.encoder.encode_named_subset(obs, subset, grad=grad) for subset in self.augmented_subsets]
        return z_full, z_subsets

    def update_critic(
        self,
        state,
        z_full,
        z_subsets,
        action,
        reward,
        discount,
        next_state,
        next_z_full,
        encoder_reg_loss: torch.Tensor | None = None,
        update_encoder: bool = False,
    ):
        metrics = {}
        with torch.no_grad():
            dist = self.actor(next_state, next_z_full)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_state, next_z_full, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)
            self._ensure_finite("target_Q", target_Q)

        z_all = torch.cat([z_full] + z_subsets, dim=0)
        repeated_state = torch.cat([state] * self.num_repetitions, dim=0)
        repeated_action = torch.cat([action] * self.num_repetitions, dim=0)
        repeated_target_Q = torch.cat([target_Q] * self.num_repetitions, dim=0)

        Q1, Q2 = self.critic(repeated_state, z_all, repeated_action)
        Q1_chunks = Q1.chunk(self.num_repetitions, dim=0)
        Q2_chunks = Q2.chunk(self.num_repetitions, dim=0)
        target_chunks = repeated_target_Q.chunk(self.num_repetitions, dim=0)

        critic_reg_loss = self.latent_aug_alpha * (
            F.mse_loss(Q1_chunks[0], target_chunks[0]) + F.mse_loss(Q2_chunks[0], target_chunks[0])
        )

        if len(z_subsets) > 0:
            aug_q1 = torch.cat(Q1_chunks[1:], dim=0)
            aug_q2 = torch.cat(Q2_chunks[1:], dim=0)
            aug_target = torch.cat(target_chunks[1:], dim=0)
            critic_aug_loss = self.latent_aug_beta * (
                F.mse_loss(aug_q1, aug_target) + F.mse_loss(aug_q2, aug_target)
            )
        else:
            critic_aug_loss = torch.tensor(0.0, device=self.device)

        critic_loss = critic_reg_loss + critic_aug_loss
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
        metrics["critic_reg_loss"] = critic_reg_loss.item()
        metrics["critic_aug_loss"] = critic_aug_loss.item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["critic_total_loss"] = total_loss.item()
        return metrics

    def update_actor(self, state, z_full, z_subsets):
        metrics = {}
        actor_obs = torch.cat([z_full] + z_subsets, dim=0)
        repeated_state = torch.cat([state] * self.num_repetitions, dim=0)
        critic_obs = torch.cat([z_full] * self.num_repetitions, dim=0)

        dist = self.actor(repeated_state, actor_obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(repeated_state, critic_obs, action)
        Q = torch.min(Q1, Q2)

        log_prob_chunks = log_prob.chunk(self.num_repetitions, dim=0)
        q_chunks = Q.chunk(self.num_repetitions, dim=0)

        actor_loss_reg = (self.alpha.detach() * log_prob_chunks[0] - q_chunks[0]).mean()
        if len(z_subsets) > 0:
            actor_loss_aug = (
                self.alpha.detach() * torch.cat(log_prob_chunks[1:], dim=0) - torch.cat(q_chunks[1:], dim=0)
            ).mean()
        else:
            actor_loss_aug = torch.tensor(0.0, device=self.device)
        actor_loss = (self.latent_aug_alpha * actor_loss_reg) + (self.latent_aug_beta * actor_loss_aug)
        self._ensure_finite("actor_loss", actor_loss)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        metrics["actor_grad_norm"] = self._module_grad_norm(self.actor)
        self.actor_opt.step()

        alpha_loss_reg = (self.alpha * (-log_prob_chunks[0] - self.target_entropy).detach()).mean()
        if len(z_subsets) > 0:
            alpha_loss_aug = (
                self.alpha
                * (-torch.cat(log_prob_chunks[1:], dim=0) - self.target_entropy).detach()
            ).mean()
        else:
            alpha_loss_aug = torch.tensor(0.0, device=self.device)
        alpha_loss = (self.latent_aug_alpha * alpha_loss_reg) + (self.latent_aug_beta * alpha_loss_aug)

        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics["actor_loss_reg"] = actor_loss_reg.item()
        metrics["actor_loss_aug"] = actor_loss_aug.item()
        metrics["actor_loss"] = actor_loss.item()
        metrics["actor_entropy"] = -log_prob.mean().item()
        metrics["actor_target_entropy"] = self.target_entropy
        metrics["alpha_loss_reg"] = alpha_loss_reg.item()
        metrics["alpha_loss_aug"] = alpha_loss_aug.item()
        metrics["alpha_loss"] = alpha_loss.item()
        metrics["alpha_value"] = self.alpha.item()
        return metrics

    def update(self, replay_iter, step):
        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        state, obs, action, reward, discount, next_state, next_obs = utils.to_torch(batch, self.device)
        encoder_update_active = self._encoder_update_active(step)
        self._set_encoder_update_mode(encoder_update_active)

        if encoder_update_active:
            z_full, z_subsets = self._encode_training_latents(obs, grad=True)
            next_z_full = self.encoder.encode_named_subset(next_obs, self.encoder.all_cameras, grad=False)
        else:
            with utils.eval_mode(self.encoder):
                z_full, z_subsets = self._encode_training_latents(obs, grad=False)
                next_z_full = self.encoder.encode_named_subset(next_obs, self.encoder.all_cameras, grad=False)

        anchor_reg_loss = torch.tensor(0.0, device=self.device)
        subset_align_reg_loss = torch.tensor(0.0, device=self.device)
        total_reg_loss = torch.tensor(0.0, device=self.device)
        if encoder_update_active:
            if self.vit_anchor_reg_weight > 0.0:
                anchor_reg_loss = self._compute_anchor_reg(obs, self.encoder.all_cameras, z_full)
                total_reg_loss = total_reg_loss + (self.vit_anchor_reg_weight * anchor_reg_loss)
            if z_subsets and self.vit_subset_alignment_reg_weight > 0.0:
                subset_align_reg_loss = torch.stack([F.mse_loss(z_subset, z_full) for z_subset in z_subsets]).mean()
                total_reg_loss = total_reg_loss + (
                    self.vit_subset_alignment_reg_weight * subset_align_reg_loss
                )

        self._ensure_finite("latent_z_full", z_full)
        self._ensure_finite("latent_next_z_full", next_z_full)
        metrics["batch_reward"] = reward.mean().item()
        metrics["encoder_trainable"] = float(self.encoder_trainable)
        metrics["encoder_update_active"] = float(encoder_update_active)
        metrics["encoder_anchor_reg_loss"] = float(anchor_reg_loss.item())
        metrics["encoder_subset_align_reg_loss"] = float(subset_align_reg_loss.item())
        metrics["encoder_total_reg_loss"] = float(total_reg_loss.item())
        metrics["latent_num_aug_subsets"] = len(self.augmented_subsets)
        self._record_latent_metrics(metrics, "latent_z", z_full)
        self._record_latent_metrics(metrics, "latent_next_z", next_z_full)
        if z_subsets:
            subset_cat = torch.cat(z_subsets, dim=0)
            self._ensure_finite("latent_z_subset", subset_cat)
            self._record_latent_metrics(metrics, "latent_subset_z", subset_cat)
            subset_mses = [F.mse_loss(z_subset, z_full) for z_subset in z_subsets]
            subset_coses = [F.cosine_similarity(z_subset, z_full, dim=-1).mean() for z_subset in z_subsets]
            metrics["latent_subset_to_full_mse"] = float(torch.stack(subset_mses).mean().item())
            metrics["latent_subset_to_full_cos"] = float(torch.stack(subset_coses).mean().item())
        metrics.update(
            self.update_critic(
                state,
                z_full,
                z_subsets,
                action,
                reward,
                discount,
                next_state,
                next_z_full,
                encoder_reg_loss=total_reg_loss,
                update_encoder=encoder_update_active,
            )
        )
        metrics.update(self.update_actor(state.detach(), z_full.detach(), [z.detach() for z in z_subsets]))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
