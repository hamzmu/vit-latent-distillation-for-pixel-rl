# LATENT SELF-DISTILLATION FOR MULTI-CAMERA MASKED AUTOENCODING WITH VISION TRANSFORMERS

This repository is the working codebase for the paper **“Latent Self-Distillation for Multi-Camera Masked Autoencoding with Vision Transformers.”** It contains both stages of the project:

1. **ViT pretraining**, where we learn a multiview visual encoder from multi-camera observations using masked reconstruction and latent-alignment objectives.
2. **Policy training**, where we use the pretrained encoder inside downstream reinforcement learning on Meta-World robotic manipulation tasks.

The code is organized around the main claim of the paper: a multi-camera encoder should learn a shared latent space in which partial camera observations remain compatible with the full multi-camera latent, so that a policy trained with all cameras can still function when only a subset of views is available at test time.

## Repository structure

- `MultiModalViT/`: core code for ViT pretraining, evaluation, figures, and paper notes.
- `MultiModalViT/seq_training.py`: sequential launcher for the original three pretraining methods: `joint`, `distill`, and `curriculum`.
- `MultiModalViT/seq_training2.py`: sequential launcher for the tuned MAD-camera joint-family pretraining ablations used for downstream policy transfer.
- `MultiModalViT/mad/`: downstream policy training code built around the Meta-World experiments.
- `MultiModalViT/mad/seq_training.py`: sequential launcher for policy-training sweeps with pretrained ViT latents.
- `paper_details/`: supporting training notes and scheduling details.

## What this codebase does

At a high level, the project runs in two stages.

### Stage 1: Visual pretraining

We pretrain a Vision Transformer-based masked autoencoder on synchronized observations from three cameras. The main pretraining objectives in this repository are:

- **full-view reconstruction**: reconstruct masked observations when all cameras are available
- **single-view or subset reconstruction**: reconstruct the full multiview scene from only one visible camera or a camera subset
- **latent alignment / self-distillation**: force the subset latent to stay close to the full multiview latent

The core pretraining question is: **which objective produces the best shared latent space for downstream control under missing-camera evaluation?**

### Stage 2: Downstream policy training

We then load a pretrained ViT checkpoint into a reinforcement-learning agent for Meta-World. The policy is trained on the `button-press-topdown` task with three cameras:

- `first -> gripperPOV`
- `third1 -> corner2`
- `third2 -> corner3`

Evaluation is performed on:

- `first` only
- `third1` only
- `third2` only
- all three cameras together

This is the downstream robustness test for the paper: **does pretraining produce a latent space that still supports control when cameras are removed at evaluation time?**

## Environment setup

The main environment file used in this repository is:

- `MultiModalViT/vsmae_env.yml`

Create and activate the environment:

```bash
cd /home/medcvr/Desktop/hamza/multicam-vit
conda env create -f MultiModalViT/vsmae_env.yml
conda activate multi-vit
```

The current project code assumes a working PyTorch + CUDA + MuJoCo + Meta-World installation. In our runs, the launchers also work with an explicit Python path if needed via `--python_exe`.

## Stage 1: ViT pretraining

There are two main ways to run pretraining in this repo.

### Option A: Compare the three original pretraining methods

This is the original pretraining comparison used to compare:

- `joint`
- `distill`
- `curriculum`

Run:

```bash
cd /home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT
python seq_training.py
```

By default, this launcher:

- uses cameras `gripperPOV`, `corner`, `corner2`
- trains each method for `50,000` steps
- writes outputs under `MultiModalViT/seq_runs/`

What each method means:

- `joint`: full-view reconstruction, single-view reconstruction, and latent alignment in one training loop
- `distill`: two-stage teacher-student training
- `curriculum`: curriculum-based multi-camera MAE training

### Option B: Run the tuned MAD-camera pretraining sweep

This is the pretraining launcher used for the downstream policy-transfer experiments in `mad/`.

Run:

```bash
cd /home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT
python seq_training2.py
```

By default, this launcher:

- uses cameras `gripperPOV`, `corner2`, `corner3`
- uses MAD camera aliases `first`, `third1`, `third2`
- trains three joint-family ablations
- writes outputs under `MultiModalViT/seq_runs2/`

The tuned checkpoint currently used by default for policy training is:

- `seq_runs2/joint_distill_mse_w1p25_20260404_233618/vtmae_joint_distill_mse_w1p25_50000.pt`

### Pretraining outputs

Typical pretraining outputs include:

- checkpoints
- training logs
- preview reconstructions
- evaluation folders comparing reconstruction and latent alignment

Important directories:

- `MultiModalViT/seq_runs/`
- `MultiModalViT/seq_runs2/`
- `MultiModalViT/recon_compare_*`
- `MultiModalViT/eval_seq_runs2_*`

## Stage 2: Policy training from pretrained ViT latents

The downstream policy experiments live in:

- `MultiModalViT/mad/`

The current policy work in this repository focuses on `vit_latent_subset_sac`, which uses a pretrained ViT encoder and trains SAC on both full and subset latents.

### Run the current policy sweep

```bash
cd /home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT/mad
python seq_training.py
```

This launcher currently runs a sequential sweep of unfrozen subset-SAC policy variants. It uses:

- task: `button-press-topdown`
- cameras: `[first, third1, third2]`
- eval cameras: `[[first],[third1],[third2],[first,third1,third2]]`
- batch size: `32`
- evaluation every `10,000` steps
- `20` evaluation episodes per camera condition

Outputs are written under:

- `MultiModalViT/mad/seq_runs/`

Each run directory typically contains:

- `logs/train.log`
- `logs/train_1.csv`
- `logs/eval_1.csv`
- evaluation videos if enabled

### Run a single policy training job directly

If you want to run a single policy job rather than the sweep launcher:

```bash
cd /home/medcvr/Desktop/hamza/multicam-vit/MultiModalViT/mad
python train.py \
  agent=vit_latent_subset_sac \
  task=button-press-topdown \
  cameras=[first,third1,third2] \
  eval_cameras=[[first],[third1],[third2],[first,third1,third2]] \
  camera_alias_profile=mad \
  metaworld_backend=ours \
  vit_checkpoint=../seq_runs2/joint_distill_mse_w1p25_20260404_233618/vtmae_joint_distill_mse_w1p25_50000.pt
```

The main downstream configuration lives in:

- `MultiModalViT/mad/config.yaml`

## Recommended workflow

If you want to reproduce the main project workflow, use this order:

1. Create the conda environment from `MultiModalViT/vsmae_env.yml`.
2. Run ViT pretraining with `MultiModalViT/seq_training.py` if you want the original `joint/distill/curriculum` comparison.
3. Run `MultiModalViT/seq_training2.py` if you want the tuned MAD-camera joint-family sweep used for policy transfer.
4. Choose a pretrained checkpoint from `MultiModalViT/seq_runs2/`.
5. Run policy training in `MultiModalViT/mad/seq_training.py`.
6. Compare full-camera and single-camera success in the `eval_1.csv` outputs.

## What to read next

Useful project notes already in the repository:

- `MultiModalViT/VIT_PRETRAINING_METHODS_AND_RESULTS.md`
- `MultiModalViT/MAD_POLICY_TRAINING_RESULTS.md`
- `MultiModalViT/ENVIRONMENT_AND_SETUP.md`
- `paper_details/README_3CAM_TRAINING_DETAILS.md`
- `paper_details/README_3CAM_SCHEDULING.md`

## Summary

This repository is not just a generic Vision Transformer project or a generic robot-RL project. It is specifically the codebase for studying the following pipeline:

- learn a shared multiview visual latent with masked autoencoding and self-distillation
- test whether that latent transfers into robust control under missing-camera evaluation

The pretraining stage lives in `MultiModalViT/`, and the downstream policy-training stage lives in `MultiModalViT/mad/`.
