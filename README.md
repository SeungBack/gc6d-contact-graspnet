# Contact-GraspNet Baseline for GraspClutter6D

**GraspClutter6D** (RA-L 2025) · [[Paper]](https://arxiv.org/abs/2504.06866) [[Website]](https://sites.google.com/view/graspclutter6d) [[Dataset]](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) [[Video]](https://youtu.be/NkKkfVS5wZ4)

| Repository | Description |
|---|---|
| [graspclutter6dAPI](https://github.com/SeungBack/graspclutter6dAPI) | Dataset toolkit — annotation loading, grasp evaluation |
| [gc6d-pose-anno](https://github.com/SeungBack/gc6d-pose-anno) | 6D object pose annotation tool (BOP format) |
| **gc6d-contact-graspnet** | 6-DoF grasp detection baseline **(this repo)** |

---

## Overview

This repository provides a PyTorch implementation of the **Contact-GraspNet baseline used in the GraspClutter6D paper**. It includes preprocessing, training, inference, and evaluation workflows for GraspClutter6D. GraspNet-1Billion support is also included for comparison and cross-dataset experiments.

- GraspClutter6D training and evaluation
- Single-camera and multi-camera GraspClutter6D training (`realsense-d415`, `realsense-d435`, `azure-kinect`, `zivid`)
- GraspNet-1Billion training/evaluation for comparison experiments
- Contact-point label preprocessing from raw grasp annotations
- GPU-based collision detection post-processing
- [ ] TODO: Add metrics for GraspClutter6D / GraspNet-1B
- [ ] TODO: Simulated grasping evaluation code (PyBullet)

---

## Requirements

- Tested on Python 3.10, PyTorch 2.5.1, CUDA 12.1

## Installation

```bash
# 1. Create conda environment
conda create -n cgnet python=3.10
conda activate cgnet

# 2. Install PyTorch (CUDA 12.1 example)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install dataset APIs
pip install git+https://github.com/SeungBack/graspnetAPI.git
pip install git+https://github.com/SeungBack/graspclutter6dAPI.git

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Pretrained Weights

Download pretrained checkpoints from Google Drive and place them under `ckpts/`:

```
ckpts/
├── train_gc6d_rs.pth     # trained on GraspClutter6D (realsense-d435)
└── train_g1b_rs.pth      # trained on GraspNet-1Billion (realsense)
```

| Model | Role | Trained on | Download |
|-------|------|------------|----------|
| `train_gc6d_rs.pth` | GC6D checkpoint | GraspClutter6D (realsense-d435) | [Google Drive](https://drive.google.com/file/d/1-RTDg_1GlvOAMRIkd9aSXmLubaloFvsa/view?usp=sharing) |
| `train_g1b_rs.pth` | G1B checkpoint | GraspNet-1Billion (realsense) | [Google Drive](https://drive.google.com/file/d/1biNXRIZ6V--ivLIYGgjojXIzrCiJeZXl/view?usp=sharing) |

---

## Data Preparation

Dataset roots are read from environment variables by default:

```bash
export GC6D_ROOT=/path/to/GraspClutter6D
export G1B_ROOT=/path/to/GraspNet-1Billion   # optional unless using G1B training/eval
```

You can also edit `DATA_PATH` directly in `configs/datasets/graspclutter6d.yaml` or `configs/datasets/graspnet1b.yaml`.

### GraspClutter6D

1. Download GraspClutter6D from the [project page](https://sites.google.com/view/graspclutter6d) or [Hugging Face](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D).

2. Preprocess contact labels for the target camera:

```bash
# Single-camera preprocessing
python scripts/preprocess_gc6d.py --camera realsense-d435 --split train

# Multi-camera preprocessing
for cam in realsense-d415 realsense-d435 azure-kinect zivid; do
  python scripts/preprocess_gc6d.py --camera $cam --split train &
done
wait
```

### GraspNet-1Billion

Use this section if you want to train on GraspNet-1Billion or run G1B evaluation.

1. Download GraspNet-1Billion from [graspnet.net](https://graspnet.net/).

2. Preprocess contact labels:

```bash
# Single worker
python scripts/preprocess_g1b.py --camera realsense --split train

# Parallel preprocessing (N workers, worker IDs 0..N-1)
for i in $(seq 0 $((N-1))); do
  python scripts/preprocess_g1b.py \
      --camera realsense --split train \
      --n_workers $N --worker_id $i &
done
wait
```

---

## Training

### GraspClutter6D

```bash
# Single-camera training (realsense-d435)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_gc6d.yaml \
    --exp_name train_gc6d

# Multi-camera training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_gc6d_allcam.yaml \
    --exp_name train_gc6d_allcam
```

### GraspNet-1Billion Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_g1b.yaml \
    --exp_name train_g1b
```

Checkpoints are saved under `experiments/<config_name>/<config_parent>/<exp_name>/`
(for example, `experiments/train_gc6d/configs/train_gc6d/ckpt-last.pth`).

---

## Evaluation

`--grasp_root` defaults to `$GC6D_ROOT` or `$G1B_ROOT` depending on `--test_dataset`, and `--dump_dir` defaults to `dump/<config_name>_<test_dataset>`.

Valid cameras:

- `gc6d`: `realsense-d415`, `realsense-d435`, `azure-kinect`, `zivid`
- `g1b`: `realsense`, `kinect`

GC6D evaluation currently supports `--split test`. For cross-dataset evaluation, set the env var of the **test** dataset or pass `--grasp_root` explicitly.

### GraspClutter6D Evaluation

```bash
# train GraspClutter6D -> test GraspClutter6D
python test.py \
    --model_config configs/train_gc6d.yaml \
    --model_checkpoint ckpts/train_gc6d_rs.pth \
    --test_dataset gc6d --camera realsense-d435 \
    --split test --infer --eval --collision_thresh 0.01

# train GraspNet-1Billion -> test GraspClutter6D (cross-dataset)
python test.py \
    --model_config configs/train_g1b.yaml \
    --model_checkpoint ckpts/train_g1b_rs.pth \
    --test_dataset gc6d --camera realsense-d435 \
    --split test --infer --eval --collision_thresh 0.01
```

### GraspNet-1Billion Evaluation

```bash
# train GraspNet-1Billion -> test GraspNet-1Billion
python test.py \
    --model_config configs/train_g1b.yaml \
    --model_checkpoint ckpts/train_g1b_rs.pth \
    --test_dataset g1b --camera realsense \
    --split test --infer --eval --collision_thresh 0.01
```

```bash
# train GraspClutter6D -> test GraspNet-1Billion (cross-dataset)
python test.py \
    --model_config configs/train_gc6d.yaml \
    --model_checkpoint ckpts/train_gc6d_rs.pth \
    --test_dataset g1b --camera realsense \
    --split test --infer --eval --collision_thresh 0.01
```


## Repository Structure

```
├── train.py                          # Training entry point
├── test.py                           # Inference + evaluation entry point
├── models/
│   ├── cgnet.py                      # ContactGraspNet model (PointNet++ backbone)
│   ├── losses.py                     # Contact grasp losses
│   └── pointnet2_utils.py
├── data/
│   ├── gc6d.py                       # GraspClutter6D data loader
│   └── g1b.py                        # GraspNet-1Billion data loader
├── utils/
│   ├── collision_detector.py         # CPU + GPU collision detection
│   ├── runner.py                     # Training loop
│   └── ...
├── configs/
│   ├── train_gc6d.yaml               # Training config for GraspClutter6D (single cam)
│   ├── train_gc6d_allcam.yaml        # Training config for GraspClutter6D (4 cams)
│   ├── train_g1b.yaml                # Training config for GraspNet-1Billion
│   ├── model_params.yaml             # Model hyperparameters
│   └── datasets/
│       ├── graspclutter6d.yaml       # GraspClutter6D dataset config (set DATA_PATH)
│       └── graspnet1b.yaml           # GraspNet-1B dataset config (set DATA_PATH)
└── scripts/
    ├── preprocess_gc6d.py            # Contact label preprocessing for GraspClutter6D
    └── preprocess_g1b.py             # Contact label preprocessing for GraspNet-1B
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{back2025graspclutter6d,
  title={Graspclutter6d: A large-scale real-world dataset for robust perception and grasping in cluttered scenes},
  author={Back, Seunghyeok and Lee, Joosoon and Kim, Kangmin and Rho, Heeseon and Lee, Geonhyup and Kang, Raeyoung and Lee, Sangbeom and Noh, Sangjun and Lee, Youngjin and Lee, Taeyeop and others},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}

@inproceedings{sundermeyer2021contact,
  title={Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE international conference on robotics and automation (ICRA)},
  pages={13438--13444},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgments

This codebase is built on top of [contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch) by Ethan Chun, a PyTorch re-implementation of [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) (Sundermeyer et al., ICRA 2021).
