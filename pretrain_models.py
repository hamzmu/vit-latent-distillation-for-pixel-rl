# =========================
# pretrain_models.py
# (NO segmentation anywhere)
# =========================
from __future__ import annotations

from vit_pytorch.vit import pair, Transformer
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D


class EarlyCNN(nn.Module):
    def __init__(self, in_channels: int, encoder_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, encoder_dim // 8, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(encoder_dim // 8, encoder_dim // 4, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(encoder_dim // 4, encoder_dim // 2, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(encoder_dim // 2, encoder_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class VST(nn.Module):
    """
    Two-stream ViT encoder scaffold:
      - primary image stream (required)
      - auxiliary image stream (optional; e.g., second camera)
    """

    def __init__(
        self,
        *,
        image_size,
        aux_size,
        image_patch_size,
        aux_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        image_channels: int = 3,
        aux_channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        frame_stack: int = 1,
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        aux_height, aux_width = pair(aux_size)
        image_patch_height, image_patch_width = pair(image_patch_size)
        aux_patch_height, aux_patch_width = pair(aux_patch_size)

        # sizes
        self.image_height = image_height
        self.image_width = image_width
        self.aux_height = aux_height
        self.aux_width = aux_width

        # patch sizes
        self.image_patch_height = image_patch_height
        self.image_patch_width = image_patch_width
        self.aux_patch_height = aux_patch_height
        self.aux_patch_width = aux_patch_width

        self.image_channels = image_channels
        self.aux_channels = aux_channels
        self.frame_stack = frame_stack

        assert image_height % image_patch_height == 0 and image_width % image_patch_width == 0, \
            "Image dimensions must be divisible by the patch size."
        assert aux_height % aux_patch_height == 0 and aux_width % aux_patch_width == 0, \
            "Aux dimensions must be divisible by the patch size."

        # counts (for pos_embed shape only)
        num_patches_image = (image_height // image_patch_height) * (image_width // image_patch_width)
        num_patches_aux = (aux_height // aux_patch_height) * (aux_width // aux_patch_width)
        num_patches = num_patches_image + num_patches_aux

        image_patch_dim = image_channels * image_patch_height * image_patch_width
        aux_patch_dim = aux_channels * aux_patch_height * aux_patch_width

        # patch embedding paths (used to form MAE targets and/or tokens)
        self.image_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=image_patch_height, p2=image_patch_width),
            nn.LayerNorm(image_patch_dim),
            nn.Linear(image_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.aux_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=aux_patch_height, p2=aux_patch_width),
            nn.LayerNorm(aux_patch_dim),
            nn.Linear(aux_patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # kept for shape introspection elsewhere
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()


class VSMAE(nn.Module):
    """
    Vision + Aux-Image MAE with separate randomized masking per modality.
    - image is REQUIRED (x["image"])
    - aux image is OPTIONAL (x.get("image_aux"))
    Both streams reconstruct with MSE on raw patch pixels.
    """

    def __init__(
        self,
        *,
        encoder: VST,
        decoder_dim: int,
        masking_ratio_a: float = 0.75,
        masking_ratio_b: float = 0.75,
        decoder_depth: int = 1,
        decoder_heads: int = 8,
        decoder_dim_head: int = 64,
        use_sincosmod_encodings: bool = True,
        frame_stack: int = 1,
        auxloss_multiplier: float = 1.0,
    ):
        super().__init__()
        self.masking_ratio_a = masking_ratio_a
        self.masking_ratio_b = masking_ratio_b

        self.auxloss_multiplier = auxloss_multiplier
        self.frame_stack = frame_stack
        self.use_sincosmod_encodings = use_sincosmod_encodings

        # encoder (ViT)
        self.encoder: VST = encoder
        _, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # patch grids
        Hp = self.encoder.image_height // self.encoder.image_patch_height
        Wp = self.encoder.image_width // self.encoder.image_patch_width
        Ha = self.encoder.aux_height // self.encoder.aux_patch_height
        Wa = self.encoder.aux_width // self.encoder.aux_patch_width

        self.Hp, self.Wp = Hp, Wp
        self.Ha, self.Wa = Ha, Wa

        # Early-CNN tokenizers + pooling to patch grid
        img_ch = self.encoder.image_channels
        aux_ch = self.encoder.aux_channels
        self.early_conv_vision = EarlyCNN(img_ch, encoder_dim)
        self.early_conv_aux = EarlyCNN(aux_ch, encoder_dim)
        self.cnn_pool_img = nn.AdaptiveAvgPool2d((Hp, Wp))
        self.cnn_pool_aux = nn.AdaptiveAvgPool2d((Ha, Wa))
        print("VT-MAE: Early-CNN tokeniser (pooled to patch grid) — aux optional")

        # retain patch→emb layers for MAE targets (we use raw patch pixels as targets)
        self.image_to_patch = encoder.image_to_patch_embedding[0]
        pixel_values_per_patch = encoder.image_to_patch_embedding[2].weight.shape[-1]

        self.aux_to_patch = encoder.aux_to_patch_embedding[0]
        aux_values_per_patch = encoder.aux_to_patch_embedding[2].weight.shape[-1]

        # decoder + projections
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(decoder_dim, decoder_depth, decoder_heads, decoder_dim_head, decoder_dim * 4)

        # heads (both are pixel regression heads)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.to_aux_pixels = nn.Linear(decoder_dim, aux_values_per_patch)

        # fixed 2D sin-cos encodings on patch grids
        enc_pe2d = PositionalEncoding2D(encoder_dim)
        dec_pe2d = PositionalEncoding2D(decoder_dim)

        img_pe_enc = enc_pe2d(torch.zeros(1, Hp, Wp, encoder_dim)).flatten(1, 2)  # [1, Hp*Wp, D]
        aux_pe_enc = enc_pe2d(torch.zeros(1, Ha, Wa, encoder_dim)).flatten(1, 2)  # [1, Ha*Wa, D]

        img_pe_dec = dec_pe2d(torch.zeros(1, Hp, Wp, decoder_dim)).flatten(1, 2)  # [1, Hp*Wp, Dd]
        aux_pe_dec = dec_pe2d(torch.zeros(1, Ha, Wa, decoder_dim)).flatten(1, 2)  # [1, Ha*Wa, Dd]

        self.register_buffer("image_enc_pos_embedding", img_pe_enc)
        self.register_buffer("aux_enc_pos_embedding", aux_pe_enc)
        self.register_buffer("image_dec_pos_embedding", img_pe_dec)
        self.register_buffer("aux_dec_pos_embedding", aux_pe_dec)

        # modality embeddings: 0=image, 1=aux
        self.encoder_modality_embedding = nn.Embedding(2, encoder_dim)
        self.decoder_modality_embedding = nn.Embedding(2, decoder_dim)

        # projection on pooled rep (for get_embeddings)
        self.repr_ln = nn.LayerNorm(self.encoder_dim)

    # -------------- token helpers --------------
    def _tokens_from_image(self, x_img: torch.Tensor) -> torch.Tensor:
        feat = self.early_conv_vision(x_img)   # [B, D, Hc, Wc]
        feat = self.cnn_pool_img(feat)         # [B, D, Hp, Wp]
        return feat.flatten(2).transpose(1, 2) # [B, Hp*Wp, D]

    def _tokens_from_aux_optional(
        self,
        x_aux: torch.Tensor | None,
        B: int,
        device: torch.device,
        use_aux: bool,
    ) -> torch.Tensor:
        if (not use_aux) or (x_aux is None):
            return torch.zeros((B, 0, self.encoder_dim), device=device)
        feat = self.early_conv_aux(x_aux)      # [B, D, Hc, Wc]
        feat = self.cnn_pool_aux(feat)         # [B, D, Ha, Wa]
        return feat.flatten(2).transpose(1, 2) # [B, Ha*Wa, D]

    def forward(self, x: dict, *, use_vision: bool = True, use_aux: bool = True, **kwargs):
        """
        Expected x:
          x["image"]:      [B, Ci, H, W]
          x["image_aux"]:  [B, Ca, Ha, Wa] (optional)
        """
        assert "image" in x, "Image input is required."
        device = x["image"].device
        B = x["image"].shape[0]

        x_aux = x.get("image_aux", None)
        use_aux = bool(use_aux and (x_aux is not None) and isinstance(x_aux, torch.Tensor) and x_aux.numel() > 0)

        # BUG A FIX: correct raw patch dims when modality disabled
        img_patch_dim = self.encoder.image_channels * self.encoder.image_patch_height * self.encoder.image_patch_width
        aux_patch_dim = self.encoder.aux_channels * self.encoder.aux_patch_height * self.encoder.aux_patch_width

        img_patches = self.image_to_patch(x["image"]) if use_vision else torch.zeros((B, 0, img_patch_dim), device=device)
        Ni = img_patches.shape[1]

        if use_aux:
            aux_patches = self.aux_to_patch(x_aux)
            Na = aux_patches.shape[1]
        else:
            aux_patches = torch.zeros((B, 0, aux_patch_dim), device=device)
            Na = 0

        img_tok = self._tokens_from_image(x["image"]) if use_vision else torch.zeros((B, 0, self.encoder_dim), device=device)
        aux_tok = self._tokens_from_aux_optional(x_aux, B, device, use_aux)

        if self.use_sincosmod_encodings and img_tok.shape[1] > 0:
            img_tok = img_tok + self.encoder_modality_embedding.weight[0] + self.image_enc_pos_embedding
        if self.use_sincosmod_encodings and aux_tok.shape[1] > 0:
            aux_tok = aux_tok + self.encoder_modality_embedding.weight[1] + self.aux_enc_pos_embedding

        tokens = torch.cat([img_tok, aux_tok], dim=1)  # [B, Ni+Na, D]
        Nt = tokens.shape[1]

        # independent masking per modality
        if Ni > 0:
            n_mask_img = int(self.masking_ratio_a * Ni)
            perm_img = torch.rand(B, Ni, device=device).argsort(dim=-1)
            m_idx_img = perm_img[:, :n_mask_img]
            u_idx_img = perm_img[:, n_mask_img:]
        else:
            m_idx_img = torch.zeros((B, 0), dtype=torch.long, device=device)
            u_idx_img = torch.zeros((B, 0), dtype=torch.long, device=device)

        if Na > 0:
            n_mask_aux = int(self.masking_ratio_b * Na)
            perm_aux = torch.rand(B, Na, device=device).argsort(dim=-1)
            m_idx_aux = perm_aux[:, :n_mask_aux]
            u_idx_aux = perm_aux[:, n_mask_aux:]
        else:
            m_idx_aux = torch.zeros((B, 0), dtype=torch.long, device=device)
            u_idx_aux = torch.zeros((B, 0), dtype=torch.long, device=device)

        u_idx = torch.cat([u_idx_img, u_idx_aux + Ni], dim=1)
        m_idx = torch.cat([m_idx_img, m_idx_aux + Ni], dim=1)
        batch_range = torch.arange(B, device=device)[:, None]

        enc_in = tokens[batch_range, u_idx]        # [B, Nu, D]
        enc_out = self.encoder.transformer(enc_in) # [B, Nu, D]
        dec_tok = self.enc_to_dec(enc_out)         # [B, Nu, Dd]

        dec_tokens = torch.zeros(B, Nt, self.decoder_dim, device=device)
        dec_tokens[batch_range, u_idx] = dec_tok
        dec_tokens[batch_range, m_idx] = self.mask_token

        if self.use_sincosmod_encodings:
            if Ni > 0:
                dec_tokens[:, :Ni] += self.decoder_modality_embedding.weight[0]
                dec_tokens[:, :Ni] += self.image_dec_pos_embedding
            if Na > 0:
                dec_tokens[:, Ni:] += self.decoder_modality_embedding.weight[1]
                dec_tokens[:, Ni:] += self.aux_dec_pos_embedding

        dec_tokens = self.decoder(dec_tokens)  # [B, Nt, Dd]

        recon_loss_total = torch.tensor(0.0, device=device)
        recon_loss_img = torch.tensor(0.0, device=device)
        recon_loss_aux = torch.tensor(0.0, device=device)

        if m_idx_img.shape[1] > 0:
            pred_px = self.to_pixels(dec_tokens[batch_range, m_idx[:, :m_idx_img.shape[1]]])
            tgt_px = img_patches[batch_range, m_idx_img]
            recon_loss_img = F.mse_loss(pred_px, tgt_px)
            recon_loss_total = recon_loss_total + recon_loss_img

        if m_idx_aux.shape[1] > 0:
            aux_slice = slice(m_idx_img.shape[1], m_idx_img.shape[1] + m_idx_aux.shape[1])
            pred_aux = self.to_aux_pixels(dec_tokens[batch_range, m_idx[:, aux_slice]])
            tgt_aux = aux_patches[batch_range, m_idx_aux]
            recon_loss_aux = F.mse_loss(pred_aux, tgt_aux)
            recon_loss_total = recon_loss_total + self.auxloss_multiplier * recon_loss_aux

        debug = {"Ni": Ni, "Na": Na, "m_idx_img": m_idx_img, "m_idx_aux": m_idx_aux}

        if kwargs.get("return_debug", False):
            return recon_loss_total, debug

        if kwargs.get("return_breakdown", False):
            return {
                "total": recon_loss_total,
                "rgb_mse": recon_loss_img,
                "aux_mse": recon_loss_aux,
                "rgb_weight": 1.0,
                "aux_weight": float(self.auxloss_multiplier),
                "n_img_masked": int(m_idx_img.shape[1]),
                "n_aux_masked": int(m_idx_aux.shape[1]),
            }

        return recon_loss_total

    def get_embeddings(self, x: dict, eval: bool = True, use_vision: bool = True, use_aux: bool = True):
        assert "image" in x, "Image input is required."
        self.eval() if eval else self.train()

        device = x["image"].device
        B = x["image"].shape[0]

        x_aux = x.get("image_aux", None)
        use_aux = bool(use_aux and (x_aux is not None) and isinstance(x_aux, torch.Tensor) and x_aux.numel() > 0)

        img_tok = self._tokens_from_image(x["image"]) if use_vision else torch.zeros((B, 0, self.encoder_dim), device=device)
        aux_tok = self._tokens_from_aux_optional(x_aux, B, device, use_aux)

        if self.use_sincosmod_encodings and img_tok.shape[1] > 0:
            img_tok = img_tok + self.encoder_modality_embedding.weight[0] + self.image_enc_pos_embedding
        if self.use_sincosmod_encodings and aux_tok.shape[1] > 0:
            aux_tok = aux_tok + self.encoder_modality_embedding.weight[1] + self.aux_enc_pos_embedding

        tokens = torch.cat([img_tok, aux_tok], dim=1)
        encoded_tokens = self.encoder.transformer(tokens)
        rep = self.repr_ln(encoded_tokens.mean(dim=1))
        return rep
