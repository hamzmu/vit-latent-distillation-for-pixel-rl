# =========================
# pretrain_models.py
# ViT-only (NO CNN anywhere)
# 3-view masked autoencoder with dynamic per-sample view subsets
# =========================
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from vit_pytorch.vit import Transformer, pair
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D


class MultiViewVST(nn.Module):
    """
    Shared-patch-embedding ViT encoder for fixed multi-view RGB input.
    Designed for exactly 3 cameras by default, but supports any num_views >= 1.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels: int = 3,
        num_views: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        frame_stack: int = 1,
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            "Image dimensions must be divisible by patch size."
        )
        assert num_views >= 1, "num_views must be >= 1"

        # Keep these attribute names for compatibility with existing scripts.
        self.image_height = image_height
        self.image_width = image_width
        self.image_patch_height = patch_height
        self.image_patch_width = patch_width
        self.image_channels = channels
        self.frame_stack = frame_stack
        self.num_views = num_views

        self.num_patches_per_view = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Shared patch embedding across all views.
        self.to_patch = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_height,
            p2=patch_width,
        )
        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Kept for shape introspection with existing utilities.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches_per_view + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def patchify_views(self, views: torch.Tensor) -> torch.Tensor:
        """
        views: [B, V, C, H, W] -> patches: [B, V, N, P]
        """
        assert views.ndim == 5, f"Expected [B,V,C,H,W], got {tuple(views.shape)}"
        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected V={self.num_views}, got {V}"
        assert C == self.image_channels, f"Expected C={self.image_channels}, got {C}"
        assert (H, W) == (self.image_height, self.image_width), (
            f"Expected spatial {(self.image_height, self.image_width)}, got {(H, W)}"
        )

        flat = views.reshape(B * V, C, H, W)
        patches = self.to_patch(flat)
        return patches.reshape(B, V, self.num_patches_per_view, -1)

    def embed_view_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [B, V, N, P] -> tokens: [B, V, N, D]
        """
        B, V, N, P = patches.shape
        tokens = self.patch_to_embedding(patches.reshape(B * V * N, P)).reshape(B, V, N, -1)
        return tokens

    def embed_views(self, views: torch.Tensor) -> torch.Tensor:
        patches = self.patchify_views(views)
        return self.embed_view_patches(patches)


class MultiViewVSMAE(nn.Module):
    """
    Multi-view ViT MAE with dynamic per-sample view visibility and mask ratios.

    Forward expects:
      views: [B, V, C, H, W]
      visible_views: [B, V] bool  (dropped views have no encoder tokens)
      mask_ratios: [B, V] float in [0,1]

    Reconstruction always predicts all views for all patches; loss is computed only
    on masked patches, where dropped views are fully masked by construction.
    """

    def __init__(
        self,
        *,
        encoder: MultiViewVST,
        decoder_dim: int,
        decoder_depth: int = 1,
        decoder_heads: int = 8,
        decoder_dim_head: int = 64,
        num_views: int = 3,
        use_sincosmod_encodings: bool = True,
        frame_stack: int = 1,
    ):
        super().__init__()

        assert num_views == encoder.num_views, (
            f"encoder.num_views ({encoder.num_views}) must match num_views ({num_views})"
        )

        self.encoder: MultiViewVST = encoder
        self.encoder_dim = encoder.pos_embedding.shape[-1]
        self.decoder_dim = decoder_dim
        self.num_views = num_views
        self.frame_stack = frame_stack
        self.use_sincosmod_encodings = use_sincosmod_encodings

        self.patch_height = self.encoder.image_patch_height
        self.patch_width = self.encoder.image_patch_width
        self.image_height = self.encoder.image_height
        self.image_width = self.encoder.image_width
        self.channels = self.encoder.image_channels
        self.num_patches_per_view = self.encoder.num_patches_per_view

        # Shared patchify from encoder.
        self.to_patch = self.encoder.to_patch
        patch_dim = self.channels * self.patch_height * self.patch_width

        self.enc_to_dec = nn.Linear(self.encoder_dim, decoder_dim) if self.encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(decoder_dim, decoder_depth, decoder_heads, decoder_dim_head, decoder_dim * 4)

        # ModuleList keeps per-view heads explicit and extensible.
        self.to_view_pixels = nn.ModuleList([nn.Linear(decoder_dim, patch_dim) for _ in range(num_views)])

        # Fixed 2D positional encodings on the patch grid.
        Hp = self.image_height // self.patch_height
        Wp = self.image_width // self.patch_width

        enc_pe2d = PositionalEncoding2D(self.encoder_dim)
        dec_pe2d = PositionalEncoding2D(decoder_dim)

        patch_enc = enc_pe2d(torch.zeros(1, Hp, Wp, self.encoder_dim)).flatten(1, 2)  # [1,N,D]
        patch_dec = dec_pe2d(torch.zeros(1, Hp, Wp, decoder_dim)).flatten(1, 2)        # [1,N,Dd]
        self.register_buffer("patch_enc_pos_embedding", patch_enc)
        self.register_buffer("patch_dec_pos_embedding", patch_dec)

        # Learned camera-ID embeddings.
        self.encoder_camera_embedding = nn.Embedding(num_views, self.encoder_dim)
        self.decoder_camera_embedding = nn.Embedding(num_views, self.decoder_dim)

        self.repr_ln = nn.LayerNorm(self.encoder_dim)

        print("VT-MAE: Multi-view ViT patch tokeniser (3-view dynamic masking)")

    def _validate_inputs(
        self,
        views: torch.Tensor,
        visible_views: torch.Tensor | None,
        mask_ratios: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert views.ndim == 5, f"Expected [B,V,C,H,W], got {tuple(views.shape)}"
        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected V={self.num_views}, got V={V}"
        assert C == self.channels, f"Expected channels={self.channels}, got channels={C}"
        assert (H, W) == (self.image_height, self.image_width), (
            f"Expected {(self.image_height, self.image_width)}, got {(H, W)}"
        )

        device = views.device

        if visible_views is None:
            visible_views = torch.ones((B, V), dtype=torch.bool, device=device)
        else:
            visible_views = visible_views.to(device=device, dtype=torch.bool)
            assert visible_views.shape == (B, V), (
                f"visible_views must be [B,V], got {tuple(visible_views.shape)}"
            )

        # Every sample must keep at least one visible camera.
        assert bool((visible_views.sum(dim=1) >= 1).all()), "Each sample must have at least one visible camera."

        if mask_ratios is None:
            mask_ratios = torch.full((B, V), 0.75, device=device, dtype=views.dtype)
            mask_ratios = torch.where(visible_views, mask_ratios, torch.ones_like(mask_ratios))
        else:
            mask_ratios = mask_ratios.to(device=device, dtype=views.dtype)
            assert mask_ratios.shape == (B, V), f"mask_ratios must be [B,V], got {tuple(mask_ratios.shape)}"

        mask_ratios = mask_ratios.clamp(0.0, 1.0)
        # Dropped views are always fully masked.
        mask_ratios = torch.where(visible_views, mask_ratios, torch.ones_like(mask_ratios))

        return views, visible_views, mask_ratios

    def _add_encodings(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B,V,N,D] -> [B,V,N,D] with camera + position encodings.
        """
        B, V, N, D = tokens.shape
        out = tokens
        if self.use_sincosmod_encodings:
            cam_ids = torch.arange(V, device=tokens.device)
            cam_embed = self.encoder_camera_embedding(cam_ids).view(1, V, 1, D)
            pos_embed = self.patch_enc_pos_embedding[:, :N, :].view(1, 1, N, D)
            out = out + cam_embed + pos_embed
        return out

    def _add_decoder_encodings(self, dec_tokens: torch.Tensor) -> torch.Tensor:
        """
        dec_tokens: [B,V,N,Dd] -> [B,V,N,Dd]
        """
        B, V, N, Dd = dec_tokens.shape
        out = dec_tokens
        if self.use_sincosmod_encodings:
            cam_ids = torch.arange(V, device=dec_tokens.device)
            cam_embed = self.decoder_camera_embedding(cam_ids).view(1, V, 1, Dd)
            pos_embed = self.patch_dec_pos_embedding[:, :N, :].view(1, 1, N, Dd)
            out = out + cam_embed + pos_embed
        return out

    def _sample_keep_mask(self, visible_views: torch.Tensor, mask_ratios: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Returns keep mask [B,V,N] where True means token is provided to encoder.
        """
        B, V = visible_views.shape
        device = visible_views.device

        # Rank patches per sample/view and keep lowest ranks up to keep_count.
        rand = torch.rand((B, V, num_patches), device=device)
        ranks = rand.argsort(dim=-1).argsort(dim=-1)

        keep_counts = ((1.0 - mask_ratios) * num_patches).round().long().clamp(0, num_patches)
        keep_mask = ranks < keep_counts[..., None]
        keep_mask = keep_mask & visible_views[..., None]

        # Ensure at least one encoder token per sample.
        keep_per_sample = keep_mask.sum(dim=(1, 2))
        zero_keep_idx = torch.where(keep_per_sample == 0)[0]
        for b in zero_keep_idx.tolist():
            visible = torch.where(visible_views[b])[0]
            assert visible.numel() > 0, "No visible cameras for sample."
            keep_mask[b, int(visible[0]), 0] = True

        return keep_mask

    def _encode_with_variable_keep(self, tokens: torch.Tensor, keep_mask: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode only kept tokens with no fake/padded dropped-view tokens.

        tokens: [B,V,N,D]
        keep_mask: [B,V,N] bool

        Returns:
          - encoded_tokens_full: [B,V,N,D] with zeros where token was not encoded
          - keep_indices: list of B tensors containing flattened kept indices in [0, V*N)
        """
        B, V, N, D = tokens.shape
        device = tokens.device
        flat_tokens = tokens.reshape(B, V * N, D)
        flat_keep = keep_mask.reshape(B, V * N)

        keep_indices = [torch.where(flat_keep[b])[0] for b in range(B)]

        encoded_full = torch.zeros((B, V * N, self.encoder_dim), device=device, dtype=tokens.dtype)

        # Group samples by token count to preserve true "no dropped tokens" while
        # still batching transformer calls where possible.
        groups: Dict[int, List[int]] = {}
        for b, idx in enumerate(keep_indices):
            groups.setdefault(int(idx.numel()), []).append(b)

        for tok_count, sample_ids in groups.items():
            assert tok_count > 0, "Every sample must have at least one kept token."
            batch_tokens = torch.stack([flat_tokens[b, keep_indices[b]] for b in sample_ids], dim=0)  # [G,L,D]
            batch_encoded = self.encoder.transformer(batch_tokens)
            for row, b in enumerate(sample_ids):
                encoded_full[b, keep_indices[b]] = batch_encoded[row]

        return encoded_full.reshape(B, V, N, self.encoder_dim), keep_indices

    def _decode_all_patches(self, encoded_full: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        encoded_full: [B,V,N,D]
        keep_mask: [B,V,N]
        Returns predicted patches for every view/patch: [B,V,N,P]
        """
        B, V, N, D = encoded_full.shape
        device = encoded_full.device

        dec_full = self.enc_to_dec(encoded_full.reshape(B, V * N, D)).reshape(B, V, N, self.decoder_dim)

        # Start with mask token everywhere, then fill kept positions with encoded tokens.
        mask_token = self.mask_token.view(1, 1, 1, self.decoder_dim)
        dec_tokens = mask_token.expand(B, V, N, self.decoder_dim).clone()
        dec_tokens[keep_mask] = dec_full[keep_mask]

        dec_tokens = self._add_decoder_encodings(dec_tokens)
        dec_out = self.decoder(dec_tokens.reshape(B, V * N, self.decoder_dim)).reshape(B, V, N, self.decoder_dim)

        preds = [self.to_view_pixels[v](dec_out[:, v]) for v in range(V)]
        pred_patches = torch.stack(preds, dim=1)
        return pred_patches

    def _compute_losses(
        self,
        pred_patches: torch.Tensor,
        tgt_patches: torch.Tensor,
        keep_mask: torch.Tensor,
        visible_views: torch.Tensor,
        pattern_weights: torch.Tensor | None,
        cross_view_loss_weight: float,
    ) -> dict:
        """
        Returns loss dictionary with per-view and pattern-aware components.
        """
        B, V, N, P = pred_patches.shape
        device = pred_patches.device

        # Loss is only on masked patches (dropped views are fully masked).
        loss_mask = ~keep_mask  # [B,V,N]
        mse_patch = (pred_patches - tgt_patches).pow(2).mean(dim=-1)  # [B,V,N]

        masked_counts = loss_mask.sum(dim=-1).clamp(min=1)  # [B,V]
        sample_view_loss = (mse_patch * loss_mask.float()).sum(dim=-1) / masked_counts.float()  # [B,V]

        dropped_views = ~visible_views
        dropped_float = dropped_views.float()
        visible_float = visible_views.float()

        visible_denom = visible_float.sum(dim=1).clamp(min=1.0)
        cross_denom = dropped_float.sum(dim=1).clamp(min=1.0)

        visible_loss_per_sample = (sample_view_loss * visible_float).sum(dim=1) / visible_denom
        cross_loss_per_sample = (sample_view_loss * dropped_float).sum(dim=1) / cross_denom
        cross_loss_per_sample = torch.where(
            dropped_float.sum(dim=1) > 0,
            cross_loss_per_sample,
            torch.zeros_like(cross_loss_per_sample),
        )

        cross_weight_matrix = torch.where(
            dropped_views,
            torch.full_like(sample_view_loss, float(cross_view_loss_weight)),
            torch.ones_like(sample_view_loss),
        )
        sample_recon_loss = (sample_view_loss * cross_weight_matrix).mean(dim=1)

        if pattern_weights is None:
            pattern_weights = torch.ones((B,), device=device, dtype=sample_recon_loss.dtype)
        else:
            pattern_weights = pattern_weights.to(device=device, dtype=sample_recon_loss.dtype)
            assert pattern_weights.shape == (B,), (
                f"pattern_weights must be [B], got {tuple(pattern_weights.shape)}"
            )

        weighted_sample_loss = sample_recon_loss * pattern_weights
        total_loss = weighted_sample_loss.mean()

        per_view_mse = sample_view_loss.mean(dim=0)

        return {
            "total": total_loss,
            "recon_unweighted": sample_recon_loss.mean(),
            "visible_loss": visible_loss_per_sample.mean(),
            "cross_view_loss": cross_loss_per_sample.mean(),
            "per_view_mse": per_view_mse,
            "pattern_weight_mean": pattern_weights.mean(),
            "masked_counts": masked_counts,
            "loss_mask": loss_mask,
            "sample_view_loss": sample_view_loss,
            "sample_recon_loss": sample_recon_loss,
        }

    def forward(
        self,
        views: torch.Tensor | dict,
        *,
        visible_views: torch.Tensor | None = None,
        mask_ratios: torch.Tensor | None = None,
        pattern_ids: torch.Tensor | None = None,
        pattern_weights: torch.Tensor | None = None,
        cross_view_loss_weight: float = 1.5,
        return_breakdown: bool = False,
        return_debug: bool = False,
    ):
        """
        Args:
          views: [B,V,C,H,W] (preferred) or dict with key "views"
        """
        if isinstance(views, dict):
            if "views" in views:
                views = views["views"]
            else:
                raise AssertionError("Expected dict input with key 'views'.")

        views, visible_views, mask_ratios = self._validate_inputs(views, visible_views, mask_ratios)

        # Targets are always all views.
        tgt_patches = self.encoder.patchify_views(views)

        tokens = self.encoder.embed_view_patches(tgt_patches)
        tokens = self._add_encodings(tokens)

        keep_mask = self._sample_keep_mask(visible_views, mask_ratios, self.num_patches_per_view)

        encoded_full, keep_indices = self._encode_with_variable_keep(tokens, keep_mask)
        pred_patches = self._decode_all_patches(encoded_full, keep_mask)

        out = self._compute_losses(
            pred_patches,
            tgt_patches,
            keep_mask,
            visible_views,
            pattern_weights,
            cross_view_loss_weight,
        )

        # Pattern-wise loss means (if provided) for logging.
        if pattern_ids is not None:
            pattern_ids = pattern_ids.to(device=views.device, dtype=torch.long)
            assert pattern_ids.shape[0] == views.shape[0], "pattern_ids must have length B"
            for pid, name in ((0, "triple"), (1, "single")):
                mask = pattern_ids == pid
                if bool(mask.any()):
                    out[f"pattern_loss_{name}"] = out["sample_recon_loss"][mask].mean()
                else:
                    out[f"pattern_loss_{name}"] = torch.tensor(0.0, device=views.device)

        if return_debug:
            out["debug"] = {
                "keep_mask": keep_mask,
                "loss_mask": out["loss_mask"],
                "visible_views": visible_views,
                "mask_ratios": mask_ratios,
                "pred_patches": pred_patches.detach(),
                "target_patches": tgt_patches.detach(),
                "keep_indices": keep_indices,
                "pattern_ids": pattern_ids.detach().clone() if pattern_ids is not None else None,
                "num_patches": self.num_patches_per_view,
            }

        if return_breakdown:
            out["view0_mse"] = out["per_view_mse"][0]
            out["view1_mse"] = out["per_view_mse"][1] if self.num_views > 1 else torch.tensor(0.0, device=views.device)
            out["view2_mse"] = out["per_view_mse"][2] if self.num_views > 2 else torch.tensor(0.0, device=views.device)
            return out

        return out["total"]

    def get_embeddings(
        self,
        views: torch.Tensor | dict,
        *,
        visible_views: torch.Tensor | None = None,
        eval: bool = True,
    ) -> torch.Tensor:
        """
        Returns pooled encoder representations for any camera subset.

        views: [B,V,C,H,W]
        visible_views: [B,V] bool, optional. If omitted, all views are used.
        """
        if isinstance(views, dict):
            if "views" in views:
                views = views["views"]
            else:
                raise AssertionError("Expected dict input with key 'views'.")

        self.eval() if eval else self.train()

        views, visible_views, _ = self._validate_inputs(views, visible_views, mask_ratios=None)

        patches = self.encoder.patchify_views(views)
        tokens = self.encoder.embed_view_patches(patches)
        tokens = self._add_encodings(tokens)

        keep_mask = visible_views[..., None].expand(-1, -1, self.num_patches_per_view)
        encoded_full, keep_indices = self._encode_with_variable_keep(tokens, keep_mask)

        B, V, N, D = encoded_full.shape
        flat_encoded = encoded_full.reshape(B, V * N, D)

        reps = []
        for b in range(B):
            idx = keep_indices[b]
            reps.append(self.repr_ln(flat_encoded[b, idx].mean(dim=0)))
        return torch.stack(reps, dim=0)


# Backward-compatible aliases.
VST = MultiViewVST
VSMAE = MultiViewVSMAE
