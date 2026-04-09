# ViT Latent RL Agents

This directory now contains two SAC agents that use a frozen pretrained 3-camera ViT-MAE encoder from the parent repository.

## Why add these agents

The pretrained ViT encoder already gives representation robustness:
- `z(full)` and `z(subset)` are trained to live in a compatible latent space
- the latent size is fixed across 1, 2, or 3 visible cameras

That alone does not guarantee policy robustness. A policy trained only on `z(full)` can still overfit to the full-view latent distribution and degrade when deployed on `z(single)`.

These two agents separate the two questions cleanly:

1. Does a frozen robust encoder alone transfer from full-view RL training to single-view deployment?
2. Does policy-side subset training improve deployment robustness further?

## Agent 1: `vit_latent_full_sac`

File:
- `algos/vit_latent_full_sac.py`

Motivation:
- This is the clean baseline.
- Freeze the pretrained encoder.
- Train SAC only on the full 3-camera latent during RL.
- At deployment, evaluate the same policy with whichever subset latent is available.

Interpretation:
- Encoder robustness only.
- No explicit policy robustness training.

Use this when you want to answer:
- "If the latent space is aligned well enough, can SAC trained on `z_full` transfer to `z_single` without extra RL tricks?"

## Agent 2: `vit_latent_subset_sac`

File:
- `algos/vit_latent_subset_sac.py`

Motivation:
- The encoder is already robust, but the policy may still not be.
- This agent adds policy robustness by training SAC on mixed subset latents while keeping the encoder frozen.

How it works:
- The frozen ViT encodes:
  - `z_full`
  - subset latents such as `z(c1)`, `z(c2)`, `z(c3)`
- The critic is trained on both the full latent and subset latents against the same target.
- The actor is optimized from both full and subset latents, but the sampled actions are scored against the full latent critic representation, following the stabilizing idea used in MAD.

Interpretation:
- Encoder robustness + policy robustness.

Use this when you want to answer:
- "If the policy is also exposed to subset latents during RL, does single-camera deployment improve?"

## Expected comparison

`vit_latent_full_sac`
- simplest baseline
- strongest test of whether the pretrained latent alignment alone is enough
- likely best when deploying on full views
- may degrade more under camera drop-off

`vit_latent_subset_sac`
- stronger deployment-oriented method
- should improve robustness when one or more cameras are missing
- may slightly trade off some full-view specialization for broader subset robustness

## Important usage note

The RL camera ordering must match the ordering used for ViT pretraining.

In practice:
- train RL with `cameras=[...]` in the same order used by the pretrained encoder
- keep `eval_cameras` as subsets of that same ordered set

These agents use camera identities when building ViT visibility masks, so ordering matters.

The code now resolves the default MAD camera aliases automatically for the frozen-ViT agents:
- `first -> gripperPOV`
- `third1 -> corner`
- `third2 -> corner2`

That makes the default MAD camera names compatible with the current ViT checkpoints, which were pretrained on `gripperPOV, corner, corner2`.

Meta-World RL in this repo now defaults to `metaworld_backend=ours`, which uses the same camera/render wrapper family as the ViT pretraining code. You can still switch back to the original MAD wrapper with `metaworld_backend=mad`.

## Example commands

Baseline full-latent SAC:

```bash
cd mad
python train.py \
  agent=vit_latent_full_sac \
  task=button-press-topdown \
  cameras=[first,third1,third2] \
  eval_cameras=[[first],[third1],[third2],[first,third1,third2]] \
  vit_checkpoint=../vtmae_3cam_full.pt \
  device=cuda
```

Subset-robust SAC:

```bash
cd mad
python train.py \
  agent=vit_latent_subset_sac \
  task=button-press-topdown \
  cameras=[first,third1,third2] \
  eval_cameras=[[first],[third1],[third2],[first,third1,third2]] \
  vit_checkpoint=../vtmae_3cam_full.pt \
  vit_subset_mode=full_and_singles \
  latent_aug_alpha=0.8 \
  device=cuda
```

To include double-camera subsets during RL robustness training:

```bash
python train.py agent=vit_latent_subset_sac vit_subset_mode=all_nonempty
```
