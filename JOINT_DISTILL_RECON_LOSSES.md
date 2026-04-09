# Why The Joint Training Uses Both Distillation And Reconstruction

The two losses do different jobs. They are not redundant.

## Distillation Loss

Distillation says:

- `z(single-view)` should be close to `z(full-view)`

This shapes the latent space.

What it gives you:

- subset-invariant latents
- compatibility between `z(c1)`, `z(c2)`, `z(c3)`, and `z(c1,c2,c3)`
- better transfer if downstream RL is trained on one camera configuration and deployed on another

What it does **not** guarantee:

- that the latent still contains enough scene information
- that the latent is reconstructive
- that the latent is useful rather than only numerically similar

A latent can be close in cosine or MSE while still being weak for generation or control.

## Reconstruction Loss

Reconstruction says:

- from the latent, reconstruct all 3 cameras

This makes the latent functionally meaningful.

What it gives you:

- pressure to preserve scene structure
- pressure to encode enough information from a partial view
- pressure to infer missing views from what is visible

What it does **not** guarantee:

- that the subset latent lives in the same latent space as the full-view latent

A model can reconstruct reasonably well while still placing `z(single)` and `z(full)` in different regions.

## Why Both Matter

The training objective needs two properties at the same time:

1. `z(single)` should be close to `z(full)`
2. `z(single)` should still contain enough scene information

Distillation gives you the first property.

Reconstruction gives you the second property.

So the split is:

- **distillation** = align the latent space
- **reconstruction** = make the latent informative

## What Happens If You Remove One

### If you keep only distillation

The encoder can learn to imitate the full-view latent statistically, but you lose a strong functional constraint.

Risk:

- latent becomes less reconstructive
- latent becomes less grounded in image content
- latent may be numerically aligned but semantically weak

### If you keep only reconstruction

The encoder can learn useful visual features, but `z(single)` and `z(full)` may drift apart.

Risk:

- RL trained on `z(full)` may transfer poorly to `z(single)`
- subset latents may not be compatible enough for camera drop-off deployment

## Why This Matters For RL

Downstream RL only uses `z`, so latent compatibility is important.

That is why the distillation loss matters:

- it encourages `z(single)` to match the full-view latent space

But reconstruction is still necessary because it keeps the latent grounded in the scene rather than only matching a target vector.

So for RL:

- **distillation** answers: "is this the right latent space?"
- **reconstruction** answers: "does this latent still contain enough information?"

## Why Freeze The Decoder On The Single-Camera Branch

In the `joint-decoder-frozen` variant, the decoder is only frozen for the single-camera reconstruction branch.

That means:

- full 3-camera branch: `E + D` trainable
- single-camera reconstruction branch: `E + D*`
- single-camera distillation branch: `E`

This is useful because the downstream RL policy only consumes `z`, which comes from the encoder.

The goal is:

- let the decoder learn from the strongest signal, the full 3-camera reconstruction branch
- force the encoder to do the work of making single-camera latents fit the same decoder space

If the decoder were also trainable on the single-camera branch, then part of the burden of subset robustness would be absorbed by the decoder.

That is weaker for the encoder because:

- the decoder can adapt to poor single-view latents
- the encoder is under less pressure to make `z(single)` look like a strong full-view latent

Freezing the decoder only on the single-camera branch makes the objective stricter:

- the decoder stays anchored to the full-view latent manifold
- the encoder must move the single-camera latent toward what that decoder already expects

This is closer to the downstream RL goal, where only the encoder output `z` matters.

## Why Not Freeze The Decoder Everywhere

Freezing the decoder from the very start would usually be a bad idea.

If the decoder is frozen globally during training from scratch:

- the decoder is untrained
- reconstruction targets become poor supervision
- the encoder is forced to match a bad fixed decoder

That is why the decoder should still be trainable on the full 3-camera branch.

So the intended split is:

- **full-view branch** trains the decoder
- **single-view branch** trains the encoder to fit that decoder

## Short Version Of The Decoder-Freezing Logic

- freeze the decoder on the single-camera branch because RL only uses the encoder latent
- keep the decoder trainable on the full-view branch so it still learns a good scene decoder
- this makes the single-view encoder work harder to map into the full-view latent space

## Short Version

Both are needed because they solve different problems:

- **distillation** makes subset latents compatible with the full-view latent
- **reconstruction** makes subset latents informative enough to represent the scene

Using both is usually stronger than using either one alone.
