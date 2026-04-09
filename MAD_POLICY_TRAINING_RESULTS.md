# MAD Policy Training Results

## Overview

This note summarizes the downstream policy-training stage that follows ViT pretraining. The encoder is first pretrained to align full-view and subset-view latents, and then that encoder is loaded into the RL agent for policy learning. The main downstream question is whether a policy trained with a pretrained multiview latent can still operate when only one camera is available at evaluation time.

The current policy experiments focus on the Meta-World `button-press-topdown` task and compare:

- the MAD baseline
- frozen ViT-latent baselines
- unfrozen ViT-latent subset-SAC variants with different finetuning schedules and regularization strengths

## Task and evaluation protocol

The task is `button-press-topdown`, where the robot must press the button from above. Trainisssssssng uses three synchronized camera views:

- `first` mapped to `gripperPOV`
- `third1` mapped to `corner2`
- `third2` mapped to `corner3`

Policy training is done with all three training cameras available. Evaluation is then performed under four observation settings:

1. `first` only
2. `third1` only
3. `third2` only
4. all three cameras together

Each evaluation point reports the mean success over 20 episodes for each of those four settings. This is important because the central question is not only whether the policy solves the task with all cameras present, but whether it remains usable under camera drop-off.

## Hypothesis

The working hypothesis has been:

1. a pretrained ViT with aligned full-view and subset-view latents should provide a better interface for transfer under missing cameras
2. a frozen encoder alone is likely not enough
3. policy-side subset training with careful encoder finetuning should improve robustness under single-camera deployment
4. overly aggressive encoder finetuning may help task success but destroy the latent geometry that the method is supposed to preserve

So the goal is not only high reward or even high full-view success. The real target is a policy that works under single-camera evaluation while still preserving the pretrained representation structure.

## Agents and comparison logic

Three downstream policy directions have been relevant:

### MAD baseline

This is the policy-only robustness baseline. It does not rely on the pretrained ViT latent. It remains the strongest reference point for overall success on this task.

### Frozen ViT-latent baselines

These runs keep the pretrained encoder fixed and train the policy on top of that latent. The motivation is to test whether encoder robustness alone is already enough for transfer.

### Unfrozen ViT-latent subset-SAC

These runs allow the encoder to be updated during RL while also training on subset latents. This is the main method family of interest because it tries to combine:

- pretrained latent alignment
- policy robustness
- limited encoder adaptation to the downstream task

## Policy hyperparameters we changed

The policy sweeps have mostly varied a small set of ViT-finetuning and subset-training hyperparameters.

### `vit_finetune_encoder`

This is the biggest switch in the entire study.

- `false`: keep the pretrained encoder frozen
- `true`: allow RL to update the encoder

Motivation:

- test whether the pretrained latent is already sufficient
- then test whether encoder adaptation during RL is necessary

What we learned:

- frozen is too weak on this task
- unfrozen is necessary, but it introduces a strong stability tradeoff

### `vit_encoder_lr`

This controls how aggressively RL rewrites the encoder.

Values tested so far:

- `1e-5`
- `5e-6`
- `3e-6`
- `2.5e-6`
- `2e-6`

Motivation:

- `1e-5` was the aggressive setting used to test whether full adaptation could unlock strong task performance
- lower values were introduced after seeing severe latent drift and exploding critic/encoder statistics
- `2.5e-6` and `2e-6` were later midpoint / conservative tests to see whether we could keep enough adaptation while reducing collapse

What we learned:

- `1e-5` gives the best raw success, but breaks the latent space
- `5e-6` and `3e-6` can still learn, but need a good schedule
- `2e-6` and `2.5e-6` tend to become too conservative on this task

### `vit_unfreeze_after_steps`

This delays when encoder gradients are allowed to update the pretrained ViT.

Values tested so far:

- immediate finetuning
- `25,000`
- `5,000`
- `2,000`
- `1,000`

Motivation:

- large delays were introduced to let the actor and critic adapt to the pretrained latent before changing the encoder
- shorter delays were introduced after it became clear that `25k` was too conservative
- `1k` and `2k` were used to encourage earlier co-adaptation between policy and encoder

What we learned:

- `25k` freezes the encoder for too long and hurts policy learning
- `5k` was better than `25k`, but still often too slow
- `2k` and especially `1k` are the strongest schedules so far

### `vit_anchor_reg_weight`

This regularizes the RL-updated encoder toward a frozen anchor copy of the pretrained encoder.

Values tested so far:

- none in the earliest aggressive run
- `10.0`
- `2.0`
- `1.5`
- `1.0`
- `0.5`

Motivation:

- introduced after seeing the aggressive unfrozen run drift far away from the pretrained latent
- larger values were intended to prevent encoder collapse
- smaller values were later tested after strong regularization was clearly suppressing task learning

What we learned:

- `10.0` is too restrictive for this task
- `1.0` is the most useful range so far
- `0.5` is too loose in the current setup
- `1.5` did not help and often made optimization worse

### `vit_subset_alignment_reg_weight`

This regularizes subset latents to stay close to the full latent during RL finetuning.

Values tested so far:

- none in the earliest aggressive run
- `5.0`
- `1.0`
- `0.75`
- `0.5`
- `0.25`

Motivation:

- introduced to preserve the subset/full latent compatibility learned during pretraining
- then reduced gradually after we saw that strong alignment regularization was suppressing downstream control learning

What we learned:

- `5.0` is too strong
- `0.5` is the most workable setting so far
- `0.25` is too weak in the current regime
- `0.75` did not improve the tradeoff

### `latent_aug_alpha`

This controls how much subset exposure the policy sees relative to the full latent during subset-augmented training.

Values tested so far:

- implicit earlier default behavior before explicit tuning
- `0.8`
- `0.75`
- `0.7`
- `0.65`

Motivation:

- the initial thought was that pushing more subset exposure should improve single-camera robustness
- later sweeps reduced `latent_aug_alpha` to make training more subset-heavy
- then `0.75` was tested as a midpoint after `0.7` proved too aggressive

What we learned:

- `0.8` is still the best setting so far
- `0.7` and `0.65` made the objective too hard
- `0.75` was still not enough to recover performance

### `vit_subset_mode`

This changes which subset combinations are used during subset training.

Values tested so far:

- `full_and_singles`
- `all_nonempty`

Motivation:

- `full_and_singles` focuses on the main deployment setting of interest: single-camera drop-off
- `all_nonempty` was tested to force broader robustness across all partial-camera combinations

What we learned:

- `full_and_singles` is the better regime so far
- `all_nonempty` makes the training objective too hard at the current stage

### `batch_size` and `eval_every_steps`

These are not representation hyperparameters, but they mattered for interpreting the experiments.

Values used:

- early runs: `batch_size=256`, `eval_every_steps=25000`
- later controlled sweeps: `batch_size=32`, `eval_every_steps=10000`

Motivation:

- the shift to `32` made the frozen/unfrozen comparisons fairer and reduced one early confound
- evaluating every `10k` made it easier to see temporary peaks and later collapse

What we learned:

- the later `32`-batch, `10k`-eval setup is much better for diagnosis
- several runs peak mid-training and collapse later, which was easy to miss under 25k-step eval spacing

## Main results so far

### MAD baseline remains strongest

The MAD baseline is clearly the best performer overall on `button-press-topdown`.

In the 100k-step comparison run, MAD finishes around:

- `first = 1.0`
- `third1 = 1.0`
- `third2 = 0.9`
- `all cameras = 1.0`

In the older longer baseline, MAD reaches `1.0` success on all four evaluation settings.

This means the environment, reward shaping, and evaluation protocol are not themselves the bottleneck. The task is solvable with strong single-camera and full-camera robustness.

### Frozen ViT-latent baselines fail

The frozen ViT-latent baselines do not solve the task. In practice they remain near zero success.

This is a strong result in itself: a robust pretrained encoder is not enough. The policy still needs adaptation to the deployment distribution, and simply feeding the pretrained latent into SAC does not solve the task.

### The best raw ViT-latent policy run is the aggressive unfrozen run

The best raw subset-SAC result so far is the aggressive unfrozen run with immediate finetuning and `vit_encoder_lr = 1e-5`. Its final success is:

- `first = 0.85`
- `third1 = 0.70`
- `third2 = 0.55`
- `all cameras = 1.00`

This is the strongest downstream result achieved by the ViT-latent method family so far.

However, it comes with severe latent drift:

- subset-to-full latent MSE becomes very large
- cosine similarity to the original full latent falls sharply
- critic loss and encoder gradient norms become extremely large

So this run appears to solve the task partly by rewriting the pretrained representation. It is useful as an upper bound on task success, but not as the most thesis-aligned method.

### The best controlled run is the 1k warmup light-regularization variant

Among the more controlled unfrozen runs, the strongest result so far is the `warmup1k_lightreg` setting. Its final success is:

- `first = 0.90`
- `third1 = 0.10`
- `third2 = 0.00`
- `all cameras = 0.75`

This is much weaker than MAD and weaker than the aggressive unfrozen run, but the latent geometry is substantially healthier. The subset-to-full latent mismatch stays far smaller than in the aggressive run.

This run is currently the best compromise between downstream performance and representation preservation.

## What the task-specific results tell us

A very consistent pattern appears across the policy runs:

- `first` is the easiest single-camera evaluation condition
- `third1` is harder
- `third2` is usually the hardest

That matters because the all-camera success can look moderately good even when the single-camera deployment story is still weak. In the best controlled run, the all-camera case reaches `0.75`, but the two third-person single-camera settings remain poor.

So the current policy does not yet show robust transfer across all single-camera conditions. Instead, it appears to be relying most heavily on the gripper view and only partially generalizing to the harder third-person views.

## Detailed ablation history

The ablation path has been informative because each sweep was motivated by a concrete failure mode from the previous one.

### Pilot stage: frozen/full-latent baselines

The earliest policy tests asked whether the pretrained encoder alone was sufficient.

`vit_latent_full_sac`

- encoder frozen
- policy trained only on the full latent
- result: no learning

`vit_latent_subset_sac` frozen

- encoder frozen
- subset-aware policy training
- result: still no learning

Conclusion:

- the pretrained encoder is not enough by itself
- policy-side subset training alone is not enough if the encoder never adapts

There was also an early unfrozen subset-SAC pilot before the later controlled sweeps. That run briefly reached a moderate all-view average success around `0.275`, but collapsed back to zero by the final checkpoint. This was an early sign that the method could learn something useful, but that stability over the full training horizon was going to be a central problem.

### Stage 1: aggressive unfrozen finetuning

Main run:

- `vit_encoder_lr = 1e-5`
- immediate encoder finetuning
- no delayed unfreeze
- no anchor regularization
- no subset-alignment regularization
- early evaluation every `25k`
- frozen comparison in that stage used a larger batch size, so the first frozen/unfrozen contrast was not perfectly clean

Motivation:

- before adding any safeguards, we needed to know whether letting RL directly adapt the encoder would solve the task at all

Result:

- strongest raw subset-SAC result so far
- final success `0.85 / 0.70 / 0.55 / 1.00`

Failure mode:

- `latent_subset_to_full_mse ≈ 0.341`
- `latent_subset_to_full_cos ≈ 0.698`
- `critic_loss ≈ 1294`
- `encoder_grad_norm ≈ 57k`

Interpretation:

- the method can solve the task much better when unconstrained
- but it does so by heavily rewriting the encoder
- this run supports task feasibility, not the representation-preserving thesis

### Stage 2: delayed unfreeze with strong regularization

Main run:

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 25000`
- `vit_anchor_reg_weight = 10.0`
- `vit_subset_alignment_reg_weight = 5.0`
- `batch_size = 32`
- `eval_every_steps = 10000`

Motivation:

- the aggressive run showed that the encoder was drifting too far
- the next idea was to let the actor/critic warm up on a fixed pretrained latent and then only later allow gentle finetuning
- strong anchor and alignment losses were introduced to explicitly preserve the pretrained representation

Result:

- much healthier latent metrics
- but much worse policy performance
- best all-view average only `0.20`, final `0.1125`

Interpretation:

- this fixed the wrong problem too strongly
- the representation was protected, but the policy no longer adapted enough to solve the task

### Stage 3: earlier unfreeze sweeps

The next sweep tested whether earlier co-adaptation would help while keeping some regularization.

#### `warmup2k_lightreg`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 2000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`

Motivation:

- start encoder adaptation much earlier than `25k`
- reduce regularization to keep the policy from being over-constrained

Result:

- best and final all-view average `0.35`
- final single-camera average `0.3167`
- final success `0.65 / 0.10 / 0.20 / 0.45`

Interpretation:

- this was the first strong sign that earlier co-adaptation was the right direction

#### `warmup5k_balanced`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 5000`
- `vit_anchor_reg_weight = 2.0`
- `vit_subset_alignment_reg_weight = 1.0`

Motivation:

- test a middle ground between `25k` and `2k`
- keep more regularization than the light setting

Result:

- best all-view average `0.0625`
- final all-view average `0.0625`
- critic loss exploded late

Interpretation:

- still too conservative for this task

#### `warmup5k_fasterlr`

- `vit_encoder_lr = 5e-6`
- `vit_unfreeze_after_steps = 5000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`

Motivation:

- recover some of the aggressive run’s adaptation speed without fully abandoning regularization

Result:

- best all-view average `0.3125` at `50k`
- final all-view average `0.0125`
- latent mismatch became much worse than the light-reg run

Interpretation:

- this setting can learn quickly
- but it is too unstable over the full run

### Stage 4: move from 2k to 1k and make subset training explicit

After the 2k/5k sweep, the next step was to push the encoder-policy co-adaptation even earlier and make the subset augmentation settings explicit.

#### `warmup1k_lightreg`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 1000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`
- `latent_aug_alpha = 0.8`
- `vit_subset_mode = full_and_singles`

Motivation:

- if `2k` was better than `5k`, the next obvious test was `1k`
- `full_and_singles` targets the exact deployment condition we care about
- `alpha = 0.8` keeps a strong full-view anchor while still exposing the policy to single-camera latents

Result:

- best and final all-view average `0.4375`
- final single-camera average `0.3333`
- final success `0.90 / 0.10 / 0.00 / 0.75`

Interpretation:

- this is the best controlled run so far
- it still underperforms on `third1` and `third2`, but it gave the clearest success/stability tradeoff

#### `warmup2k_lighterreg`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 2000`
- `vit_anchor_reg_weight = 0.5`
- `vit_subset_alignment_reg_weight = 0.25`
- `latent_aug_alpha = 0.8`
- `vit_subset_mode = full_and_singles`

Motivation:

- test whether `warmup1k_lightreg` was still over-regularized

Result:

- best all-view average `0.20`
- final all-view average `0.0`
- latent mismatch got noticeably worse

Interpretation:

- the regularization in the winning run was not the main bottleneck
- loosening it further made the run less stable

#### `warmup2k_subsetheavy_allnonempty`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 2000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`
- `latent_aug_alpha = 0.65`
- `vit_subset_mode = all_nonempty`

Motivation:

- we wanted to push harder on subset robustness
- this run exposed the critic/actor to all non-empty camera subsets, not just singles

Result:

- almost no learning
- best all-view average only `0.0125`
- run ended early in practice

Interpretation:

- this objective is too hard at the current stage

### Stage 5: tighter 1k sweeps around the current winner

After `warmup1k_lightreg`, we made smaller local changes around that best controlled configuration.

#### `warmup1k_lowerlr`

- `vit_encoder_lr = 2e-6`
- `vit_unfreeze_after_steps = 1000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`
- `latent_aug_alpha = 0.8`

Motivation:

- the best controlled run still showed drift
- lowering the LR was meant to preserve the latent while keeping the same schedule and regularization

Result:

- best all-view average `0.1625`
- final all-view average `0.0`
- latent metrics were very good

Interpretation:

- this likely under-adapted the encoder
- the representation stayed neat, but policy learning was too weak

#### `warmup1k_medreg`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 1000`
- `vit_anchor_reg_weight = 1.5`
- `vit_subset_alignment_reg_weight = 0.75`
- `latent_aug_alpha = 0.8`

Motivation:

- try a milder version of the strong-regularization idea, but anchored to the successful `1k` schedule

Result:

- best all-view average `0.125`
- final all-view average `0.0`
- very poor final optimization behavior

Interpretation:

- stronger regularization did not improve the tradeoff

#### `warmup1k_subsetheavier`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 1000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`
- `latent_aug_alpha = 0.7`

Motivation:

- the best controlled run was still too biased toward the `first` camera
- this run was meant to push harder on subset exposure

Result:

- best all-view average `0.0875`
- final all-view average `0.0125`

Interpretation:

- stronger subset pressure did not help the hard cameras enough to justify the loss in overall learning

### Stage 6: micro-sweep around the failed 1k variants

The latest sweep tested smaller midpoint changes after the previous regressions.

#### `warmup1k_alpha075`

- `vit_encoder_lr = 3e-6`
- `vit_unfreeze_after_steps = 1000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`
- `latent_aug_alpha = 0.75`

Motivation:

- try a midpoint between the working `0.8` and the over-aggressive `0.7`

Result:

- final and best all-view average `0.0625`
- only the all-camera evaluation showed non-zero success

Interpretation:

- moving subset pressure in that direction still hurt learning too much

#### `warmup1k_lr2p5e6`

- `vit_encoder_lr = 2.5e-6`
- `vit_unfreeze_after_steps = 1000`
- `vit_anchor_reg_weight = 1.0`
- `vit_subset_alignment_reg_weight = 0.5`
- `latent_aug_alpha = 0.8`

Motivation:

- try a midpoint between the too-weak `2e-6` and the stronger `3e-6`

Result:

- best observed all-view average `0.10` at `80k`
- best single-camera average `0.1167`

Interpretation:

- this remains too conservative relative to `3e-6`
- the latest sequence was interrupted before the third planned variant launched, so this sweep is incomplete

Overall conclusion from the chronology:

- the successful regime is narrow
- earlier unfreezing was the correct idea
- `3e-6`, `1k`, `anchor=1.0`, `subset_reg=0.5`, `alpha=0.8`, `full_and_singles` is still the best controlled reference point
- later local sweeps mostly confirmed where the method stops working, which is still useful information

## Have we achieved the hypothesis?

Not fully.

### What has been confirmed

We do have evidence for several parts of the hypothesis:

- a frozen pretrained encoder alone is not enough
- policy-side subset training matters
- careful encoder finetuning can improve downstream performance compared with frozen baselines
- the tradeoff between policy success and latent preservation is real and measurable

### What has not been achieved yet

We have not yet shown the strongest desired result:

- a ViT-latent subset-SAC policy that preserves the pretrained subset/full latent interface
- while also achieving strong and stable single-camera success
- and approaching the robustness of MAD on this task

Right now the best single-task result comes from excessive encoder drift, while the best representation-preserving result still struggles badly on `third1` and `third2`.

So the overall thesis is only partially supported by the downstream policy evidence so far.

## What we think is going wrong

Several issues are likely contributing.

### 1. The policy is exploiting shaped reward without achieving success

In many later runs, eval reward remains large while eval success stays near zero. That usually means the policy has learned to move toward the correct behavior, but not with the precision needed to reliably press the button.

### 2. The policy is still biased toward the easiest camera

The `first` camera consistently outperforms the two third-person cameras. This suggests the learned controller is not yet truly camera-robust; it is still most comfortable in the camera condition that gives the clearest action-relevant signal.

### 3. Full encoder finetuning may be too blunt

The current unfrozen runs modify the whole encoder. The results suggest this causes a hard tradeoff:

- too much freedom leads to latent collapse
- too little freedom leads to poor policy learning

This may indicate that partial finetuning or scheduled finetuning is a better direction than all-or-nothing full encoder updates.

### 4. The subset policy loss may not emphasize the hard cases enough

The current subset training does not seem to be adequately improving `third1` and `third2`. The single-camera robustness objective may need to weight hard subsets more deliberately rather than treating all subset exposures similarly.

## What we want to test next

The next experiments should stay close to the best controlled regime rather than returning to the clearly failed settings.

### 1. Anchor around the 1k warmup light-regularization setup

This is the best controlled starting point so far. Future sweeps should use it as the anchor and vary only one factor at a time.

### 2. Save or select by best evaluation checkpoint

Several runs peak at mid-training and then collapse by the final checkpoint. We should track and compare the best validation point, not only the final step.

### 3. Bias training more toward the hard single-camera cases

The biggest weakness is not full-view performance. It is the low success on the two third-person single-camera settings. Future subset training should explicitly target these cases more strongly.

### 4. Try partial encoder finetuning

Instead of updating the whole encoder, we should test:

- finetuning only the top layers
- finetuning only camera-specific embeddings or small adaptation modules
- freezing the encoder again after a short adaptation window

### 5. Separate policy optimization issues from encoder issues

Some of the later runs appear to fail even when latent metrics remain healthy. That suggests part of the problem may lie in the policy optimization itself rather than only in the encoder.

### 6. Repeat the best controlled run across multiple seeds

At the moment, most comparisons are still single-seed. Before drawing stronger claims, the best settings should be rerun across several seeds.

## Summary

So far, the policy-training story is:

- MAD is still the strongest method on `button-press-topdown`
- frozen ViT-latent baselines fail
- unfrozen ViT-latent subset-SAC is the only viable direction
- aggressive finetuning gives the best raw success but destroys the latent structure
- controlled finetuning preserves the latent better but has not yet matched the task robustness we want

The most important conclusion is that the downstream hypothesis remains open. We have shown that the pretrained ViT latent is not sufficient by itself and that policy-side subset training plus encoder finetuning is necessary. But we have not yet found a policy-training regime that simultaneously preserves the representation and delivers strong single-camera transfer on this task.
