# Paper Draft Notes

These notes are grounded in:

- [`PAPER_SECTION_OUTLINE.md`](/home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT/PAPER_SECTION_OUTLINE.md)
- ViT pretraining comparison outputs under [`eval_seq_runs2_20260404_233618_vs_prev_and_regular`](/home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT/eval_seq_runs2_20260404_233618_vs_prev_and_regular)
- MAD RL experiment logs under [`mad/seq_runs`](/home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT/mad/seq_runs)

The wording below is intentionally note-like rather than polished prose.

## Abstract Notes

- Problem:
  policies trained with full multi-camera observations often fail gracefully only at the policy level; they do not guarantee that single-camera or partial-camera observations map into the same latent interface used during training.
- Research claim:
  encoder-level subset-to-full latent alignment can reduce the train-deploy gap under camera drop-off.
- Core method:
  pretrain a multi-camera ViT-MAE with full-view reconstruction, subset-view reconstruction, and latent self-distillation from subset-view latents to full-view latents.
- Downstream method:
  use the pretrained encoder inside RL and study whether policy training on full and subset latents can retain robustness when cameras are missing.
- Main pretraining finding so far:
  the MSE-aligned joint variant with moderate single-view reconstruction weight (`new_mse_w1p25`) gives the best latent alignment, while the cosine variant with the same reconstruction weight (`new_cosine_w1p25`) gives the best reconstruction.
- Main RL finding so far:
  unconstrained unfrozen finetuning gives the highest single-task success on `button-press-topdown`, but it severely distorts the pretrained latent space; lightly regularized early-unfreeze finetuning preserves alignment much better, though with lower task success.
- Current thesis-consistent conclusion:
  there is a real tradeoff between raw task reward and preserving the subset/full latent geometry that the paper is centered on.
- Limitation to state clearly:
  current RL evidence is from a narrow setting, mainly `button-press-topdown`, and should be framed as an initial case study rather than a broad generalization.

## Introduction Notes

- Motivation:
  multi-camera observations improve scene understanding, but real deployment can lose cameras or provide only a subset of views.
- Gap in prior work:
  methods such as MAD improve policy robustness, but they do not explicitly require that `z(full)` and `z(subset)` occupy the same shared latent space.
- Central thesis:
  if subset-view and full-view observations are mapped to compatible latents, then a downstream policy trained on full-view latents should transfer more naturally to partial-view deployment.
- Framing:
  this is an encoder-robustness question, not only a policy-robustness question.
- Key intuition:
  masked autoencoding makes subset latents informative, while distillation makes them compatible with the full-view latent interface.
- Proposed story:
  1. pretrain a multiview representation that aligns full and subset camera latents
  2. use that representation for downstream RL under camera subset deployment
  3. study whether preserving latent geometry helps or conflicts with policy adaptation
- Contributions draft:
  1. a multiview ViT pretraining scheme that combines reconstruction and latent alignment for subset robustness
  2. a MAD-camera pretraining evaluation showing the tradeoff between reconstruction quality and latent compatibility
  3. an RL study showing that naive unfrozen finetuning can erase latent alignment even when task success rises
  4. logging-based analysis of encoder drift during RL finetuning

## Methodology Notes

- Representation learning setup:
  use a 3-camera ViT-MAE with camera-specific embeddings and multiview masked reconstruction.
- Full-view branch:
  encode all 3 cameras to produce a strong full-view latent `z_full`.
- Subset-view branch:
  encode single-camera or subset-camera observations to produce `z_subset`.
- Objectives:
  - full-view reconstruction loss to learn a strong multiview scene representation
  - subset-view reconstruction loss to keep partial-view latents informative
  - latent self-distillation loss to enforce `z_subset ≈ z_full`
- Interpretation of losses:
  - distillation aligns the latent space
  - reconstruction keeps the latent grounded in scene content
- Pretraining variants actually tested:
  - regular joint baseline
  - joint with raised single-view reconstruction weight and cosine distillation
  - joint with raised single-view reconstruction weight and MSE distillation
- Current chosen encoder:
  [`vtmae_joint_distill_mse_w1p25_50000.pt`](/home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT/seq_runs2/joint_distill_mse_w1p25_20260404_233618/vtmae_joint_distill_mse_w1p25_50000.pt)
- Downstream RL setup:
  use `vit_latent_subset_sac` with the pretrained encoder and evaluate on full-view plus single-camera deployment settings.
- RL finetuning question:
  whether to freeze the encoder, aggressively finetune it, or lightly regularize finetuning to preserve the pretrained subset/full structure.
- Important methodological insight from experiments:
  the downstream RL stage can undo encoder robustness unless finetuning is controlled.

\section{Experiments}

### 1. ViT Pretraining Comparison

- Goal:
  compare three practical joint-pretraining variants for camera-dropoff robustness.
- Dataset/task setting:
  Meta-World camera triplet `gripperPOV`, `corner2`, `corner3` with alias mapping `first`, `third1`, `third2`.
- Models compared:
  - `regular_joint`
  - `new_cosine_w1p25`
  - `new_mse_w1p25`
- Evaluation protocol:
  reconstruct full multiview observations from single-camera inputs and measure latent similarity between each single-camera latent and the full 3-camera latent.
- Metrics:
  reconstruction total loss, per-view reconstruction loss, cosine similarity to full latent, and MSE to full latent.
- Key finding:
  `new_mse_w1p25` is the best latent-alignment model.
- Supporting numbers:
  - `new_mse_w1p25` average over single-camera subsets:
    `z_mse_to_full ≈ 4.6e-5`
    `z_cos_to_full ≈ 0.999961`
  - `new_cosine_w1p25` average reconstruction total over single-camera subsets:
    `≈ 0.002117`
  - `regular_joint` average reconstruction total:
    `≈ 0.003378`
- Interpretation:
  MSE latent alignment plus moderate single-view reconstruction weight gives the cleanest subset-to-full latent interface, while cosine + raised single-view reconstruction gives the sharpest reconstructions.
- Current paper-facing choice:
  use `new_mse_w1p25` as the main pretrained encoder because the thesis emphasizes latent compatibility over reconstruction alone.

### 2. Downstream RL Comparison

- Goal:
  test whether the pretrained latent space supports robust policy learning and deployment under missing cameras.
- Current completed comparisons:
  - MAD baseline
  - `vit_latent_subset_sac` frozen
  - `vit_latent_subset_sac` unfrozen with aggressive finetuning
  - `vit_latent_subset_sac` unfrozen with delayed unfreeze and regularization
  - `vit_latent_subset_sac` unfrozen with early unfreeze / lighter regularization sweeps
- Evaluation cameras:
  `[first]`, `[third1]`, `[third2]`, `[first,third1,third2]`
- Metrics:
  success rate and reward per evaluation camera subset, plus encoder drift diagnostics:
  `latent_subset_to_full_mse`, `latent_subset_to_full_cos`, `encoder_grad_norm`, `critic_loss`.

#### 2.1 Current Baseline Notes

- MAD baseline remains the strongest policy result in the current local experiments.
- At 100k in the latest local MAD run, success is approximately:
  `1.0 / 1.0 / 0.9 / 1.0` for `first / third1 / third2 / all-cams`.
- Placeholder:
  [TODO: insert full MAD table and figure]
- Placeholder:
  [TODO: add MVD, MV-MWM, and any additional multi-view policy baselines]

#### 2.2 ViT-Latent Policy Notes

- Frozen `vit_latent_subset_sac` fails to solve the task in the tested setting even though latent alignment remains excellent.
- Aggressive unfrozen finetuning (`lr=1e-5`, no schedule, no regularization) achieves the best single-task success:
  `0.85 / 0.70 / 0.55 / 1.00` at 100k.
- But that same aggressive run severely damages the pretrained latent geometry:
  `latent_subset_to_full_mse ≈ 0.341`
  `latent_subset_to_full_cos ≈ 0.698`
- Delayed unfreeze with strong regularization preserves the latent much better but harms task performance too much.
- Early unfreeze with light regularization is the most promising controlled strategy so far.

#### 2.3 Latest Early-Unfreeze Sweep

- Completed runs:
  - `subset_sac_unfrozen_warmup2k_lightreg`
  - `subset_sac_unfrozen_warmup5k_balanced`
  - `subset_sac_unfrozen_warmup5k_fasterlr`
- Best run in this sweep:
  `subset_sac_unfrozen_warmup2k_lightreg`
- Final success of the best run:
  `0.65 / 0.10 / 0.20 / 0.45`
- Its latent drift remains much smaller than the aggressive unfrozen run:
  `latent_subset_to_full_mse ≈ 0.0229`
  `latent_subset_to_full_cos ≈ 0.9804`
- Interpretation:
  earlier co-adaptation helps policy learning, and light regularization preserves much more of the pretrained structure than aggressive unconstrained finetuning.
- Remaining weakness:
  single-camera performance is still uneven, especially for `third1` and `third2`.

## Results And Discussion Notes

- Main positive result:
  pretraining clearly improves latent compatibility between single-camera and full-camera observations.
- Pretraining result to emphasize:
  the MSE-aligned `w1p25` model is the best current encoder if the paper prioritizes encoder-level subset/full compatibility.
- Main RL result:
  the best reward is not currently produced by the most thesis-aligned method.
- Important tradeoff:
  - aggressive finetuning improves task success but overwrites the latent geometry
  - strong regularization preserves the latent but can suppress policy learning
  - light regularization with early unfreeze is the most promising middle ground
- This supports an important discussion point:
  encoder robustness is not automatically preserved during downstream RL, even when pretraining creates a strong aligned latent.
- Another useful discussion point:
  the strongest full-view or task-specific controller is not necessarily the strongest camera-dropoff representation.
- Current interpretation for the thesis:
  if the paper is about robust shared latent spaces, then the most relevant metric is not only final task success but also whether subset-view latents remain compatible with the full-view policy interface.
- Honest limitation:
  current RL evidence is still narrow:
  mostly one task, small number of seeds, and incomplete baseline coverage.
- Additional discussion point:
  the current experiments suggest that policy-only robustness and encoder-level robustness can conflict during finetuning, which is itself a valuable result.

## Conclusion Notes

- We can now support the following conclusion direction:
  multiview masked autoencoding plus subset-to-full latent distillation can learn a representation where single-camera and full-camera observations occupy a compatible latent space.
- Among the tested pretraining variants, MSE-based latent alignment with moderate single-view reconstruction pressure gives the strongest subset/full compatibility.
- In downstream RL, naive unfrozen finetuning can deliver high task success on a specific task but at the cost of destroying the pretrained latent geometry.
- Controlled early-unfreeze finetuning appears to be a more thesis-consistent compromise because it preserves much more of the subset/full structure while still enabling some policy adaptation.
- The broader conclusion to state carefully:
  robust multiview representation learning is necessary but not sufficient; the RL stage must also be designed to preserve the latent interface learned during pretraining.
- Final limitation statement:
  broader multi-task and multi-seed experiments are still needed before claiming strong generalization beyond the current case study.

## Placeholder Items To Fill Later

- [TODO: add exact MAD baseline table/plot for the final experiments section]
- [TODO: add other MAD-family baselines: MVD, MV-MWM, DRQ or other policy baselines if run]
- [TODO: add multi-seed statistics instead of single-seed values]
- [TODO: decide whether the paper’s main downstream method is the stable early-unfreeze variant or whether the aggressive unfrozen run is included only as an ablation]
- [TODO: add figure references for the reconstruction comparison images and policy success curves]
