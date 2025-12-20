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
    def __init__(
        self,
        *,
        image_size,
        segmentation_size,
        image_patch_size,
        segmentation_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        image_channels: int = 3,
        segmentation_channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        frame_stack: int = 1,
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        segmentation_height, segmentation_width = pair(segmentation_size)
        image_patch_height, image_patch_width = pair(image_patch_size)
        segmentation_patch_height, segmentation_patch_width = pair(segmentation_patch_size)

        # sizes
        self.image_height = image_height
        self.image_width = image_width
        self.segmentation_height = segmentation_height
        self.segmentation_width = segmentation_width

        # patch sizes
        self.image_patch_height = image_patch_height
        self.image_patch_width = image_patch_width
        self.segmentation_patch_height = segmentation_patch_height
        self.segmentation_patch_width = segmentation_patch_width

        self.image_channels = image_channels
        self.segmentation_channels = segmentation_channels
        self.frame_stack = frame_stack

        assert image_height % image_patch_height == 0 and image_width % image_patch_width == 0, \
            "Image dimensions must be divisible by the patch size."
        assert segmentation_height % segmentation_patch_height == 0 and segmentation_width % segmentation_patch_width == 0, \
            "Segmentation dimensions must be divisible by the patch size."

        # counts (for pos_embed shape only)
        num_patches_image = (image_height // image_patch_height) * (image_width // image_patch_width)
        num_patches_segmentation = (segmentation_height // segmentation_patch_height) * (segmentation_width // segmentation_patch_width)
        num_patches = num_patches_image + num_patches_segmentation

        image_patch_dim = image_channels * image_patch_height * image_patch_width
        segmentation_patch_dim = segmentation_channels * segmentation_patch_height * segmentation_patch_width

        # patch embedding paths (used to form MAE targets and/or tokens)
        self.image_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=image_patch_height, p2=image_patch_width),
            nn.LayerNorm(image_patch_dim),
            nn.Linear(image_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.segmentation_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=segmentation_patch_height, p2=segmentation_patch_width),
            nn.LayerNorm(segmentation_patch_dim),
            nn.Linear(segmentation_patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # kept for shape introspection elsewhere
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()



class VSMAE(nn.Module):
    """
    Vision+Segmentation MAE with separate randomized masking per modality.
    Image is REQUIRED; segmentation is OPTIONAL (can be omitted at train/test).
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
        segloss_multiplier: float = 1.0,
    ):
        super().__init__()
        self.masking_ratio_a = masking_ratio_a
        self.masking_ratio_b = masking_ratio_b

        self.segloss_multiplier = segloss_multiplier
        self.frame_stack = frame_stack
        self.use_sincosmod_encodings = use_sincosmod_encodings

        # encoder (ViT)
        self.encoder: VST = encoder
        _, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # patch grids
        Hp = self.encoder.image_height // self.encoder.image_patch_height
        Wp = self.encoder.image_width  // self.encoder.image_patch_width
        Hs = self.encoder.segmentation_height // self.encoder.segmentation_patch_height
        Ws = self.encoder.segmentation_width  // self.encoder.segmentation_patch_width

        self.Hp, self.Wp = Hp, Wp
        self.Hs, self.Ws = Hs, Ws

        self._shared_grid = (Hp == Hs) and (Wp == Ws) 
    
        # Early-CNN tokenizers + pooling to patch grid
        img_ch = self.encoder.image_channels
        seg_ch = self.encoder.segmentation_channels
        self.early_conv_vision = EarlyCNN(img_ch, encoder_dim)
        self.early_conv_segmentation = EarlyCNN(seg_ch, encoder_dim)
        self.cnn_pool_img = nn.AdaptiveAvgPool2d((Hp, Wp))
        self.cnn_pool_seg = nn.AdaptiveAvgPool2d((Hs, Ws))
        print("VT-MAE: Early-CNN tokeniser (pooled to patch grid) — segmentation optional")

        # retain patch→emb layers for MAE targets
        # image: we keep only the Rearrange as "to_patch" and use raw patch pixels as targets
        self.image_to_patch = encoder.image_to_patch_embedding[0]
        self.image_patch_to_emb = nn.Sequential(*encoder.image_to_patch_embedding[1:])
        pixel_values_per_patch = encoder.image_to_patch_embedding[2].weight.shape[-1]

        # segmentation: same pattern, but NO encoder.segmentation_patch_to_emb attr
        self.segmentation_to_patch = encoder.segmentation_to_patch_embedding[0]
        self.segmentation_patch_to_emb = nn.Sequential(*encoder.segmentation_to_patch_embedding[1:])
        segmentation_values_per_patch = encoder.segmentation_to_patch_embedding[2].weight.shape[-1]

        # decoder + projections
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(decoder_dim, decoder_depth, decoder_heads, decoder_dim_head, decoder_dim * 4)

        # heads
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.to_segmentations = nn.Linear(decoder_dim, segmentation_values_per_patch)

        # fixed 2D sin-cos encodings on patch grids
        enc_pe2d = PositionalEncoding2D(encoder_dim)
        dec_pe2d = PositionalEncoding2D(decoder_dim)

        img_pe_enc = enc_pe2d(torch.zeros(1, Hp, Wp, encoder_dim)).flatten(1, 2)   # [1, Hp*Wp, D]
        seg_pe_enc = enc_pe2d(torch.zeros(1, Hs, Ws, encoder_dim)).flatten(1, 2)   # [1, Hs*Ws, D]

        img_pe_dec = dec_pe2d(torch.zeros(1, Hp, Wp, decoder_dim)).flatten(1, 2)   # [1, Hp*Wp, Dd]
        seg_pe_dec = dec_pe2d(torch.zeros(1, Hs, Ws, decoder_dim)).flatten(1, 2)   # [1, Hs*Ws, Dd]

        self.register_buffer("image_enc_pos_embedding", img_pe_enc)
        self.register_buffer("segmentation_enc_pos_embedding", seg_pe_enc)
        self.register_buffer("image_dec_pos_embedding", img_pe_dec)
        self.register_buffer("segmentation_dec_pos_embedding", seg_pe_dec)

        self.encoder_modality_embedding = nn.Embedding(2, encoder_dim)
        self.decoder_modality_embedding = nn.Embedding(2, decoder_dim)

        # projection on pooled rep (for get_embeddings)
        self.repr_ln = nn.LayerNorm(self.encoder_dim)

    # -------------- token helpers --------------
    def _tokens_from_image(self, x_img: torch.Tensor) -> torch.Tensor:
        feat = self.early_conv_vision(x_img)   # [B, D, Hc, Wc]
        feat = self.cnn_pool_img(feat)         # [B, D, Hp, Wp]
        return feat.flatten(2).transpose(1, 2) # [B, Hp*Wp, D]

    def _tokens_from_seg_optional(self, seg: torch.Tensor | None, B: int, device: torch.device, use_segmentation: bool) -> torch.Tensor:
        if (not use_segmentation) or (seg is None):
            return torch.zeros((B, 0, self.encoder_dim), device=device)
        feat = self.early_conv_segmentation(seg)  # [B, D, Hc, Wc]
        feat = self.cnn_pool_seg(feat)            # [B, D, Hs, Ws]
        return feat.flatten(2).transpose(1, 2)    # [B, Hs*Ws, D]

    # -------------- MAE forward with optional segmentation --------------
    def forward(self, x: dict, *, use_vision: bool = True, use_segmentation: bool = True, **kwargs):
        assert "image" in x, "Image input is required."
        device = x["image"].device
        B = x["image"].shape[0]

        seg = x.get("segmentation", None)
        # robust check when segmentation key is missing / empty
        use_segmentation = bool(use_segmentation and (seg is not None) and isinstance(seg, torch.Tensor) and seg.numel() > 0)

        img_patches = self.image_to_patch(x["image"]) if use_vision else torch.zeros((B, 0, 3), device=device)  # [B, Nimg, Pv]
        Nimg = img_patches.shape[1]

        if use_segmentation:
            seg_patches = self.segmentation_to_patch(seg)  # [B, Nseg, Ps]
            Nseg = seg_patches.shape[1]
        else:
            seg_patches = torch.zeros((B, 0, 3), device=device)
            Nseg = 0

        img_tok = self._tokens_from_image(x["image"]) if use_vision else torch.zeros((B, 0, self.encoder_dim), device=device)
        seg_tok = self._tokens_from_seg_optional(seg, B, device, use_segmentation)

        if self.use_sincosmod_encodings and img_tok.shape[1] > 0:
            img_tok = img_tok + self.encoder_modality_embedding.weight[0] + self.image_enc_pos_embedding
        if self.use_sincosmod_encodings and seg_tok.shape[1] > 0:
            seg_tok = seg_tok + self.encoder_modality_embedding.weight[1] + self.segmentation_enc_pos_embedding

        tokens = torch.cat([img_tok, seg_tok], dim=1)  # [B, Nimg+Nseg, D]
        Nt, Ni, Ns = tokens.shape[1], img_tok.shape[1], seg_tok.shape[1]

        # independent masking per modality
        # image branch
        if Ni > 0:
            n_mask_img = int(self.masking_ratio_a * Ni)
            perm_img = torch.rand(B, Ni, device=device).argsort(dim=-1)
            m_idx_img = perm_img[:, :n_mask_img]
            u_idx_img = perm_img[:, n_mask_img:]
        else:
            m_idx_img = torch.zeros((B, 0), dtype=torch.long, device=device)
            u_idx_img = torch.zeros((B, 0), dtype=torch.long, device=device)

        # segmentation branch
        if Ns > 0:
            n_mask_seg = int(self.masking_ratio_b * Ns)
            perm_seg = torch.rand(B, Ns, device=device).argsort(dim=-1)
            m_idx_seg = perm_seg[:, :n_mask_seg]
            u_idx_seg = perm_seg[:, n_mask_seg:]
        else:
            m_idx_seg = torch.zeros((B, 0), dtype=torch.long, device=device)
            u_idx_seg = torch.zeros((B, 0), dtype=torch.long, device=device)

        # stitch indices in concatenated sequence (seg is offset by Ni)
        u_idx = torch.cat([u_idx_img, u_idx_seg + Ni], dim=1)
        m_idx = torch.cat([m_idx_img, m_idx_seg + Ni], dim=1)
        batch_range = torch.arange(B, device=device)[:, None]

        # encoder on unmasked tokens only
        enc_in = tokens[batch_range, u_idx]        # [B, Nu, D]
        enc_out = self.encoder.transformer(enc_in) # [B, Nu, D]
        dec_tok = self.enc_to_dec(enc_out)         # [B, Nu, Dd]

        # reinsert mask tokens at masked positions
        dec_tokens = torch.zeros(B, Nt, self.decoder_dim, device=device)
        dec_tokens[batch_range, u_idx] = dec_tok
        dec_tokens[batch_range, m_idx] = self.mask_token

        # add decoder modality + PE
        if self.use_sincosmod_encodings:
            if Ni > 0:
                dec_tokens[:, :Ni] += self.decoder_modality_embedding.weight[0]
                dec_tokens[:, :Ni] += self.image_dec_pos_embedding
            if Ns > 0:
                dec_tokens[:, Ni:] += self.decoder_modality_embedding.weight[1]
                dec_tokens[:, Ni:] += self.segmentation_dec_pos_embedding

        # transformer decoder over full sequence (unmasked + mask tokens)
        dec_tokens = self.decoder(dec_tokens)  # [B, Nt, Dd]

        recon_loss_total = torch.tensor(0.0, device=device)
        recon_loss_img = torch.tensor(0.0, device=device)
        recon_loss_seg = torch.tensor(0.0, device=device)

        # image reconstruction
        if m_idx_img.shape[1] > 0:
            pred_px = self.to_pixels(dec_tokens[batch_range, m_idx[:, :m_idx_img.shape[1]]])
            tgt_px  = img_patches[batch_range, m_idx_img]
            recon_loss_img = F.mse_loss(pred_px, tgt_px)
            recon_loss_total = recon_loss_total + recon_loss_img

        # segmentation reconstruction
        if m_idx_seg.shape[1] > 0:
            seg_slice = slice(m_idx_img.shape[1], m_idx_img.shape[1] + m_idx_seg.shape[1])
            pred_seg_logits = self.to_segmentations(dec_tokens[batch_range, m_idx[:, seg_slice]])
            tgt_seg  = seg_patches[batch_range, m_idx_seg]
            recon_loss_seg = F.binary_cross_entropy_with_logits(pred_seg_logits, tgt_seg)
            recon_loss_total = recon_loss_total + self.segloss_multiplier * recon_loss_seg

        # ---- DEBUG RETURN ----
        debug = {
            "Ni": Ni,
            "Ns": Ns,
            "m_idx_img": m_idx_img,
            "m_idx_seg": m_idx_seg,
        }

        if kwargs.get("return_debug", False):
            return recon_loss_total, debug


        if kwargs.get("return_breakdown", False):
            return {
                "total": recon_loss_total,            
                "rgb_mse": recon_loss_img,            
                "seg_bce": recon_loss_seg,            
                "rgb_weight": 1.0,
                "seg_weight": float(self.segloss_multiplier),
                "n_img_masked": int(m_idx_img.shape[1]),
                "n_seg_masked": int(m_idx_seg.shape[1]),
            }

        return recon_loss_total


    def get_embeddings(self, x: dict, eval: bool = True, use_vision: bool = True, use_segmentation: bool = True):
        assert "image" in x, "Image input is required."
        if eval:
            self.eval()
        else:
            self.train()

        device = x["image"].device
        B = x["image"].shape[0]
        seg = x.get("segmentation", None)
        # robust check when segmentation key is missing / empty
        use_segmentation = bool(use_segmentation and (seg is not None) and isinstance(seg, torch.Tensor) and seg.numel() > 0)

        img_tok = self._tokens_from_image(x["image"]) if use_vision else torch.zeros((B, 0, self.encoder_dim), device=device)
        seg_tok = self._tokens_from_seg_optional(seg, B, device, use_segmentation)

        if self.use_sincosmod_encodings and img_tok.shape[1] > 0:
            img_tok = img_tok + self.encoder_modality_embedding.weight[0] + self.image_enc_pos_embedding
        if self.use_sincosmod_encodings and seg_tok.shape[1] > 0:
            seg_tok = seg_tok + self.encoder_modality_embedding.weight[1] + self.segmentation_enc_pos_embedding

        tokens = torch.cat([img_tok, seg_tok], dim=1)
        encoded_tokens = self.encoder.transformer(tokens)
        rep = encoded_tokens.mean(dim=1)
        rep = self.repr_ln(rep)
        return rep


