# ViT Pretraining Methods And Results

## Project split: pretraining first, policy training second

Our work currently has two experimental stages:

1. **ViT pretraining**
   - Train a multi-camera visual encoder so that full-view and subset-view observations map to a compatible latent space.
   - This stage is evaluated with reconstruction loss and latent alignment metrics.

2. **Policy training**
   - Load the pretrained ViT checkpoint into the RL agent and train the downstream policy.
   - This stage tests whether a policy trained with the pretrained latent can still operate under camera drop-off.

This note is about the **first stage**: the ViT pretraining method comparison.

## Why compare ViT pretraining methods?

The paper goal is not only to reconstruct images well, but to learn a **shared multiview latent space** where:

- `z_full` from all cameras is informative
- `z_subset` from one camera or a subset of cameras stays close to `z_full`
- a downstream policy can reuse that same latent interface when cameras are missing

That means the pretraining comparison should look at both:

- **reconstruction quality**
- **latent alignment quality**

The comparison artifacts for this section are in:

- [`recon_compare_20260329_173506/comparison_summary.csv`](recon_compare_20260329_173506/comparison_summary.csv)
- [`recon_compare_20260329_173506/comparison_summary.txt`](recon_compare_20260329_173506/comparison_summary.txt)
- [`recon_compare_20260329_173506/method_figures/single_cam_total_reconstruction.png`](recon_compare_20260329_173506/method_figures/single_cam_total_reconstruction.png)
- [`recon_compare_20260329_173506/method_figures/single_cam_latent_alignment.png`](recon_compare_20260329_173506/method_figures/single_cam_latent_alignment.png)
- [`recon_compare_20260329_173506/method_figures/single_cam_per_view_heatmaps.png`](recon_compare_20260329_173506/method_figures/single_cam_per_view_heatmaps.png)

## ViT pretraining methods

### 1. Joint

**How it works**

- Train full-view reconstruction, single-view reconstruction, and subset-to-full latent alignment in the same loop.
- The model sees both the strong full-camera signal and weaker single-camera signal during one training process.

**Expected strength**

- Best direct pressure for learning a subset latent that is both informative and compatible with the full latent.

**Tradeoff**

- Multiple objectives are active at once, so reconstruction and alignment losses can compete.

### 2. Two-stage distill

**How it works**

- First train a full-view teacher.
- Then train the subset-view student to match the teacher latent while also reconstructing.

**Expected strength**

- Cleaner teacher-student interpretation.
- Easier to describe conceptually.

**Tradeoff**

- The student is optimized after the teacher is already fixed, so the full-view and subset-view objectives are not co-adapted in one loop.
- In our results, this method performs worst by a large margin.

### 3. Curriculum

**How it works**

- Train a 3-camera MAE with a curriculum schedule instead of explicit same-step latent distillation.
- The schedule gradually changes the masking / training condition to make the representation more robust.

**Expected strength**

- Stable reconstruction training.
- Can improve reconstruction for some subsets.

**Tradeoff**

- It does not push subset latents toward the full latent as directly as the joint method.
- In our results it is competitive on reconstruction, but weaker than joint on latent alignment.

## Experiments: pretraining comparison

### Setup

We compare the three pretrained checkpoints:

- `joint`
- `distill`
- `curriculum`

using single-camera ablations:

- `keep_cam0`
- `keep_cam1`
- `keep_cam2`

and evaluate:

- `total` reconstruction loss
- per-view reconstruction losses `v0`, `v1`, `v2`
- `z_cos_to_full`
- `z_mse_to_full`

The full-view case `keep_cam0_cam1_cam2` is perfect for all three methods, so the meaningful comparison is the **single-camera setting**.

### Overall result

The main result is:

- **Joint is the best overall method**
- **Curriculum is second**
- **Two-stage distill is clearly worst**

Averaged over the three single-camera cases from [`comparison_summary.csv`](recon_compare_20260329_173506/comparison_summary.csv):

| Method | Avg total recon | Avg cosine to full | Avg latent MSE to full |
| --- | ---: | ---: | ---: |
| Joint | `0.002046` | `0.999966` | `0.000051` |
| Curriculum | `0.002256` | `0.999873` | `0.000721` |
| Distill | `0.006377` | `0.971644` | `0.046855` |

This ranking is also reflected in:

- [`single_cam_total_reconstruction.png`](recon_compare_20260329_173506/method_figures/single_cam_total_reconstruction.png)
- [`single_cam_latent_alignment.png`](recon_compare_20260329_173506/method_figures/single_cam_latent_alignment.png)
- [`summary_rankings.csv`](recon_compare_20260329_173506/method_figures/summary_rankings.csv)

## Results and discussion

### Joint is the best compromise

Joint wins on **2 of the 3 single-camera reconstruction totals** and on **all 3 latent alignment comparisons**:

- `keep_cam1`: `total=0.001105`, `z_cos=0.999952`, `z_mse=0.000071`
- `keep_cam2`: `total=0.002052`, `z_cos=0.999976`, `z_mse=0.000036`
- `keep_cam0`: slightly behind curriculum on reconstruction (`0.002981` vs `0.002744`), but still best on latent alignment with `z_cos=0.999969` and `z_mse=0.000046`

These numbers indicate that joint training gives the strongest **shared latent interface** between full-view and single-view observations. That is the most thesis-aligned outcome because downstream policy transfer depends more directly on latent compatibility than on reconstruction alone.

Qualitatively, the subset reconstructions also stay consistently strong in:

- [`keep_cam0_compare_methods.png`](recon_compare_20260329_173506/keep_cam0_compare_methods.png)
- [`keep_cam1_compare_methods.png`](recon_compare_20260329_173506/keep_cam1_compare_methods.png)
- [`keep_cam2_compare_methods.png`](recon_compare_20260329_173506/keep_cam2_compare_methods.png)

### Curriculum is competitive on reconstruction but weaker on alignment

Curriculum gives the **best reconstruction total for `keep_cam0`**:

- `keep_cam0`: `total=0.002744`

It is also close to joint overall on average reconstruction:

- curriculum avg total `0.002256`
- joint avg total `0.002046`

But curriculum is clearly weaker on latent alignment:

- `keep_cam0`: `z_mse=0.000734` vs joint `0.000046`
- `keep_cam1`: `z_mse=0.000680` vs joint `0.000071`
- `keep_cam2`: `z_mse=0.000748` vs joint `0.000036`

So curriculum appears to help with visual reconstruction, but not as strongly with subset-to-full latent consistency.

**Interpretation:** a plausible explanation is that curriculum improves robustness through training schedule, but because it does not use the same direct subset-to-full latent matching objective as joint, it does not preserve the policy-facing latent interface as well.

### Two-stage distill underperforms badly

The two-stage distill method is much worse on both reconstruction and latent alignment.

Examples from [`comparison_summary.csv`](recon_compare_20260329_173506/comparison_summary.csv):

- `keep_cam0`: `total=0.004854`, `z_cos=0.972703`, `z_mse=0.045016`
- `keep_cam1`: `total=0.005722`, `z_cos=0.973093`, `z_mse=0.044518`
- `keep_cam2`: `total=0.008556`, `z_cos=0.969135`, `z_mse=0.051032`

Compared to joint, the latent MSE is larger by roughly three orders of magnitude:

- joint avg latent MSE: `0.000051`
- distill avg latent MSE: `0.046855`

The qualitative heatmap comparison in [`single_cam_per_view_heatmaps.png`](recon_compare_20260329_173506/method_figures/single_cam_per_view_heatmaps.png) is consistent with this: distill has the weakest missing-view recovery.

**Interpretation:** the likely issue is that the teacher-student split is too separated. The subset branch is trained to chase a fixed full-view target after the full-view stage is already learned, instead of learning both representations together in one loop. In our experiments, that produced a much weaker shared latent space.

## Conclusion from the pretraining comparison

For the original three-method ViT pretraining comparison:

- **Joint is the best overall method**
- **Curriculum is the best secondary baseline**
- **Two-stage distill should not be the main method**

This is why the follow-up pretraining sweeps moved into **joint-family ablations** rather than continuing with the two-stage distill or curriculum families.

## Relation to policy training

After this pretraining comparison, the project moves to the second stage:

- choose the best pretrained encoder
- load it into the policy agent
- evaluate whether the downstream RL policy keeps working when cameras are removed

So the role of the pretraining comparison is to answer:

> Which encoder produces the best subset-to-full latent interface before RL?

and the role of the policy experiments is to answer:

> Does that latent interface actually help transfer policy behavior under camera drop-off?

For the paper, this section should therefore come **before** the downstream RL section, and the downstream policy experiments should explicitly build on the winning pretraining family.

## Ablations: tuning the joint family for policy transfer

After the original three-method comparison, we did a second set of experiments focused only on the **joint family** on the MAD-camera setup. The motivation was simple: once joint emerged as the strongest base method, the next question was whether small changes to its loss design could further improve the subset-to-full latent interface that matters for downstream policy transfer.

The ablation comparison artifacts are in:

- [`eval_seq_runs2_20260404_233618_vs_prev_and_regular/comparison_summary.csv`](eval_seq_runs2_20260404_233618_vs_prev_and_regular/comparison_summary.csv)
- [`eval_seq_runs2_20260404_233618_vs_prev_and_regular/comparison_summary.txt`](eval_seq_runs2_20260404_233618_vs_prev_and_regular/comparison_summary.txt)
- [`eval_seq_runs2_20260404_233618_vs_prev_and_regular/keep_cam0_compare_methods.png`](eval_seq_runs2_20260404_233618_vs_prev_and_regular/keep_cam0_compare_methods.png)
- [`eval_seq_runs2_20260404_233618_vs_prev_and_regular/keep_cam1_compare_methods.png`](eval_seq_runs2_20260404_233618_vs_prev_and_regular/keep_cam1_compare_methods.png)
- [`eval_seq_runs2_20260404_233618_vs_prev_and_regular/keep_cam2_compare_methods.png`](eval_seq_runs2_20260404_233618_vs_prev_and_regular/keep_cam2_compare_methods.png)

### What we changed

The regular MAD-camera joint baseline is:

- `regular_joint`
- `distill_mode=cosine`
- `single_recon_weight=1.0`

From that baseline we tested two main knobs:

1. **Increase the single-view reconstruction weight**
   - motivation: push the encoder to keep more information in the single-camera latent instead of relying mainly on the full-view branch
   - tested as:
     - `prev_single_recon_w1p5`
     - `new_cosine_w1p25`

2. **Replace cosine latent distillation with MSE**
   - motivation: make the subset latent match the full latent more tightly in an absolute sense, not only by angle
   - tested as:
     - `prev_distill_mse`
     - `new_mse_w1p25`
     - `new_mse_w1p5`

This gave the following effective variants:

| Method | Distill mode | Single recon weight | Intuition |
| --- | --- | ---: | --- |
| `regular_joint` | cosine | `1.0` | original reference point |
| `prev_single_recon_w1p5` | cosine | `1.5` | stronger single-view pressure |
| `prev_distill_mse` | mse | `1.0` | change only the latent matching objective |
| `new_cosine_w1p25` | cosine | `1.25` | milder single-view increase |
| `new_mse_w1p25` | mse | `1.25` | combine moderate single-view increase with MSE alignment |
| `new_mse_w1p5` | mse | `1.5` | strong version of both changes |

### What happened during training

The training tails already show the main trend.

Final losses from the corresponding `train.log` files:

| Method | Final total | Full recon | Single recon | Distill |
| --- | ---: | ---: | ---: | ---: |
| `regular_joint` | `0.0046` | `0.0012` | `0.0031` | `0.0003` |
| `prev_single_recon_w1p5` | `0.0043` | `0.0008` | `0.0023` | `0.0001` |
| `prev_distill_mse` | `0.0036` | `0.0008` | `0.0026` | `0.0001` |
| `new_cosine_w1p25` | `0.0033` | `0.0007` | `0.0020` | `0.0001` |
| `new_mse_w1p25` | `0.0035` | `0.0007` | `0.0022` | `0.0000` |
| `new_mse_w1p5` | `0.0064` | `0.0010` | `0.0035` | `0.0001` |

Two observations are notable:

- Moderate changes improved the loss balance relative to `regular_joint`. Both `new_cosine_w1p25` and `new_mse_w1p25` reduced the final total loss and lowered the single-view reconstruction term.
- Pushing the single-view weight too hard was counterproductive. `new_mse_w1p5` had the worst final loss profile of the entire sweep, especially on the single-view reconstruction term.

### Evaluation result

Averaged over the three single-camera cases from [`comparison_summary.csv`](eval_seq_runs2_20260404_233618_vs_prev_and_regular/comparison_summary.csv):

| Method | Avg total recon | Avg cosine to full | Avg latent MSE to full |
| --- | ---: | ---: | ---: |
| `new_cosine_w1p25` | `0.002117` | `0.999952` | `0.000065` |
| `new_mse_w1p25` | `0.002684` | `0.999961` | `0.000046` |
| `prev_distill_mse` | `0.002704` | `0.999749` | `0.000194` |
| `prev_single_recon_w1p5` | `0.002949` | `0.999926` | `0.000105` |
| `regular_joint` | `0.003378` | `0.999729` | `0.000390` |
| `new_mse_w1p5` | `0.003888` | `0.999860` | `0.000142` |

This produced a clean split:

- **`new_cosine_w1p25` was the best reconstruction model**
- **`new_mse_w1p25` was the best latent-alignment model**

### Why the moderate variants worked best

#### `new_cosine_w1p25`: best reconstruction

`new_cosine_w1p25` was the strongest overall reconstruction variant:

- best average reconstruction total: `0.002117`
- best `keep_cam1` reconstruction: `0.001425`
- best `keep_cam2` reconstruction: `0.001776`

It did not beat `regular_joint` on `keep_cam0` reconstruction (`0.003151` vs `0.002904`), but it was better on the other two single-camera cases and much better on average.

The most likely explanation is that a **moderate** increase in single-view reconstruction pressure helped the model preserve more scene information in each single-camera latent without destabilizing the rest of training.

#### `new_mse_w1p25`: best latent interface

`new_mse_w1p25` had the best overall latent matching:

- best average cosine to full: `0.999961`
- best average latent MSE to full: `0.000046`

It was especially strong on:

- `keep_cam0`: `z_cos=0.999942`, `z_mse=0.000068`
- `keep_cam1`: `z_cos=0.999983`, `z_mse=0.000020`
- `keep_cam2`: `z_cos=0.999958`, `z_mse=0.000049`

This is the variant that best matches the paper thesis. The absolute reconstruction totals are not the very lowest, but the subset latent stays closest to the full latent, which is the property we care about most for policy transfer.

#### `new_mse_w1p5`: too much pressure

`new_mse_w1p5` is the clearest negative result:

- worst average reconstruction among the tuned variants: `0.003888`
- final training loss also rose to `0.0064`, clearly above the moderate settings

This suggests that combining MSE distillation with a stronger `single_recon_weight=1.5` over-constrained training instead of helping it.

### Interpretation

The ablations support three conclusions:

1. **The joint family was the right family to continue tuning**
   - all of the useful improvements came from modifying the joint objective, not from switching back to the curriculum or two-stage distill families

2. **Moderate single-view pressure helps**
   - `1.25` worked better than both the original `1.0` and the stronger `1.5` in the cosine setting

3. **MSE distillation is useful when paired with a moderate reconstruction weight**
   - switching to MSE alone was promising
   - the best overall latent result came from `mse + single_recon_weight=1.25`
   - but `mse + 1.5` was too aggressive

### Practical takeaway for downstream RL

This ablation study is why the final checkpoint used for policy experiments came from the `new_mse_w1p25` setting rather than from the original regular joint model or the more aggressive `1.5` variants.

In short:

- if the goal is **best reconstruction**, prefer `new_cosine_w1p25`
- if the goal is **best subset-to-full latent compatibility for policy transfer**, prefer `new_mse_w1p25`

That is the reason the downstream policy stage was built on:

- [`seq_runs2/joint_distill_mse_w1p25_20260404_233618/vtmae_joint_distill_mse_w1p25_50000.pt`](seq_runs2/joint_distill_mse_w1p25_20260404_233618/vtmae_joint_distill_mse_w1p25_50000.pt)
