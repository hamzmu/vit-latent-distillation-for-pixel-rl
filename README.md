# VSMAE - Vision-based State Masked Autoencoder

## 🧠 Overview  
This repository contains setup and training instructions for running **Sim2Real** and **Imitation Learning** experiments using the **VSMAE** (Vision-based State Masked Autoencoder) framework.  
The project leverages multimodal reinforcement learning with visual and segmentation-based representations for improved transfer and policy learning.

---

## ⚙️ Setup Instructions

### 1. Environment Setup
1. `cd MULTIMODAL_RL`  
2. `conda env create -f vsmae_env.yml`  
3. `conda activate vsmae`  
4. `pip install -e .`
5. `export MUJOCO_GL='egl'`
---

## 🚀 Running Training

You can run training using either of the following options:

### Option A — From the Root Directory
```bash
python sear/train.py agent@_global_=mars     task@_global_=metaworld_button-press-topdown-v3     hydra.run.dir=exp_local/${wandb.run_name}     use_wandb=true wandb.run_name=mw_button_press_topdown_no_seg
```

### Option B — Sequential Training Script
```bash
python seq_training.py
```

---

## 🎨 Running with Single RGB Inputs

For single RGB inputs (no segmentation), disable segmentation with the following flag:
```bash
python sear/train.py agent@_global_=mars     task@_global_=metaworld_button-press-topdown-v3     hydra.run.dir=exp_local/${wandb.run_name}     use_wandb=true wandb.run_name=mw_button_press_topdown_no_seg     add_segmentation_to_obs=False
```

---

## 📂 Key Files and Directories

| File / Directory | Description |
|------------------|--------------|
| `sear/agents/mars.py` | Core agent implementation (policy, loss, and training logic) |
| `sear/cfgs/agent/mars.yaml` | Configuration file for the MARS agent |
| `sear/train.py` | Main training script controlling training and evaluation |
| `sear/models/pretrain_models.py` | ViT / MAE model definitions used for multimodal feature encoding |

---




















IGNORE BELOW------
To train you need to have 3 files:
1) This code base
2) the sif file: 'vsmae.sif'
3) the SLURM files: 'tests.sh' and 'launch.sh'

I have emailed the SLURM file and sif file to you at XXX

within your cluster, setup a file system as follows:

vsmae_tests/
├── apptainer/
│   └── vsmae.sif
│
├── projects/
│   └── VSMAE/              
│
└── sbatch_dir/
    ├── tests.sh
    └── launch.sh


Next you want to find the path of the following which should look similar to what I have beside:

SIF_PATH=".../vsmae_tests/apptainer/vsmae.sif"
HOST_WORKDIR=".../vsmae_tests/projects/VSMAE"

Do the following afterwards:

1) Edit 'tests.sh' and on line 20 and 21, replace SIF_PATH and HOST_WORKDIR with yourown paths
2) Edit 'launch.sh' and on line 4 and 5, replace SIF_PATH and HOST_WORKDIR with yourown paths 


Finally you are ready to run these tests. Run the following:
sbatch launch.sh


Structure i'll have:

public-mars: 
* open sourced mars RL with full code implementation + yml file

private-mars:
* same as public + the following:
* evaluations, video, extras, recordings, .sh files, sif files



TODO:
* personalize to mars, repalce sear references
* remove comments and simplify code
* make sure everything works
* do a full from scratch yml setup
* make mars modular - ie accept single and double inputs, and allow vision only policy
* verify blurry vison
* rerun best hyper paramers tests
* run ted as well in the future
* merge blurry wrapper and wrapper
* update setup.py and persnalize
* personalize mars