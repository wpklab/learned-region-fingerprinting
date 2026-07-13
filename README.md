# Learned Region Selection for Generalized Source Identification of Additively Manufactured Parts

Code for the paper **"Learned Region Selection and Deep Learning for Generalized Source
Identification of Additively Manufactured Parts."**

Given an image of a 3D-printed part, the goal is to identify its **source** — which machine
(and, in related experiments, which process, material, or build-plate location) produced it —
in a way that **generalizes to part geometries and camera viewpoints never seen during
training**. Instead of feeding the whole high-resolution image to a CNN, the model *learns
which small regions of the part surface are most informative* (the manufacturing "fingerprint")
and classifies from those regions only.

<img width="1001" alt="Overview" src="https://github.com/user-attachments/assets/91fb9945-3828-49ee-8efb-f3f1bd8a3bea" />

**Model architecture**

<img width="1436" alt="Architecture" src="https://github.com/user-attachments/assets/5f7b865f-5834-4b4d-afcd-c536cb50e567" />

**Video inference**

<img width="1452" alt="Video inference" src="https://github.com/user-attachments/assets/487d79f9-31db-4d4e-b2c9-bce7ff412f7d" />

---

> ⚠️ **This repo is a research reference, not a plug-and-play package.** The training
> entrypoints are wired to a specific SLURM + Weights & Biases setup with several hardcoded
> paths and sweep IDs. Expect to edit a handful of constants (dataset name, class list,
> W&B project/sweep ID) before it runs in your environment. See
> [Adapting it to your setup](#adapting-it-to-your-setup).

---

## How the method works

The core contribution is a **differentiable "learned region selection"** front-end (called
**DPS** — Determinantal / Differentiable Patch Sampling) defined in
[`fingerprint_proposal.py`](fingerprint_proposal.py):

1. **Score.** A lightweight ResNet-18 *scorer* maps a downscaled copy of the full part image
   (e.g. 850×850) to a coarse 2-D saliency grid (e.g. 52×52).
2. **Select (differentiably).** A **perturbed top-k** operator (Gumbel-style noise + a custom
   backward pass) picks the `k` highest-scoring grid cells as a *soft, trainable* selection, so
   gradients flow back into the scorer. A noise scale `sigma` is annealed from soft →
   near-hard selection over training.
3. **Crop.** The selected cells are mapped back to the **full-resolution** image and cropped
   into `k` fixed-size patches (default **448×448**, `k = 8`).
4. **Encode.** Each patch is passed through a shared image backbone (EfficientNetV2, ConvNeXt,
   ViT, EVA-02, DINOv3, … — swappable by name).
5. **Aggregate & classify.** A small **cross-attention Transformer** pools the `k` patch
   embeddings with learnable query tokens and predicts the source class. At test time,
   predictions over the sampled patches are combined by majority/soft voting.

A convolutional-baseline variant (no learned selection — random crops + test-time voting on an
EfficientNetV2-M) is also provided for comparison.

## Repository contents

| File | Role |
|------|------|
| `fingerprint_proposal.py` | **Core method.** `DPS` (scorer + perturbed-top-k region selection + sigma annealing), `TransformerClassifier`, `FlexibleMLP`. Imported by the entrypoints. |
| `AIMS_fingerprint_dpp.py` | **Main training entrypoint** — full learned-region-selection model (DPS + backbone + Transformer). Trains on `aim_all_designs_all_views`. |
| `AIMS_fingerprint_design_sweep.py` | **Generalization experiment** — trains/evaluates with a *design-separation* split (train on some part geometries, test on held-out ones). |
| `AIMS_fingerprint_original_model.py` | **Baseline** — EfficientNetV2-M on random crops with test-time voting (no learned selection). |
| `region_fingerprint_faster.py` | Training/eval library used by `AIMS_fingerprint_dpp.py` (`train_model`, `test_model`, `initialize_model`, DDP metric gathering, early stopping, visualization). |
| `region_fingerprint_dpp.py` | Training/eval library used by `AIMS_fingerprint_design_sweep.py`. |
| `ml_models.py` | Training/eval library used by the baseline entrypoint. |
| `aims_H200.sbatch` | Example multi-node SLURM launch script (H200 GPUs, NCCL/NVLink tuning). |
| `requirements.txt` | Python dependencies. |

> **Note:** the three `*_model*.py`/`region_fingerprint_*`/`ml_models.py` libraries are
> parallel variants that evolved over the project (baseline vs. faster DPS vs. design-sweep).
> Each entrypoint imports the one it needs `as ml_models`; they are intentionally kept separate.

## The data

The models expect a standard **`torchvision.ImageFolder`** layout, with one subfolder per
source class. The primary dataset (`aim_all_designs_all_views`) is *flat* — images live directly
under each class:

```
data/aim_all_designs_all_views/
├── train/
│   ├── Stratasys450mc-1/          # one folder per printer (the class label)
│   │   ├── A22222_azimuth_0_polar_0.png
│   │   ├── A22222_azimuth_0_polar_14.png
│   │   └── ...
│   ├── Stratasys450mc-2/
│   └── ... (6 printers total)
└── val/
    └── ... (same 6 classes)
```

- **Classes:** 6 Stratasys 450mc (FDM) printers — `Stratasys450mc-1 … Stratasys450mc-6`.
- **Images:** 2550×2550 RGB PNG renders of printed parts.
- **Filename convention:** `{SERIAL}_azimuth_{A}_polar_{P}.png`, where the first 6 characters
  are the **physical-part serial** and the rest encode the **camera viewpoint** (azimuth / polar
  angle). Each printer has ~200 distinct parts (serials), each rendered from many viewpoints.

Some experiments use **re-partitioned** copies of this same image pool that add a design
sub-level and split by held-out attribute — e.g. `aim_all_designs_all_views_design_separation_20_train_10_val`
nests images by part design (`.../Stratasys450mc-1/Connector/…`) to test generalization to
unseen geometries, and `..._azimuth_separation_…` splits train/val by camera angle to test
viewpoint generalization.

See **[DATASET.md](DATASET.md)** for the full dataset family, sizes, and download instructions.

> 📦 **Dataset download:** *to be published on Kaggle — link coming soon.* See
> [DATASET.md](DATASET.md) for what will be uploaded.

## Setup

```bash
git clone https://github.com/wpklab/learned-region-fingerprinting
cd learned-region-fingerprinting

python -m venv .venv && source .venv/bin/activate     # or conda
# Install a torch build matching your CUDA version first (https://pytorch.org),
# then the rest:
pip install -r requirements.txt

# Place (or symlink) the dataset so the scripts find it at ./data/<dataset_name>
mkdir -p data
ln -s /path/to/downloaded/aim_all_designs_all_views data/aim_all_designs_all_views
```

Requirements: an NVIDIA GPU (the paper used multi-node H200s), CUDA 12.x, and a
**Weights & Biases** account — the training entrypoints run as W&B sweep agents.

## Running

Training is **distributed (PyTorch DDP over NCCL)** and driven by **W&B sweeps**. Each
entrypoint's `__main__` reads `SLURM_PROCID` / `WORLD_SIZE` / `SLURM_GPUS_ON_NODE`, initializes
the process group, and — on rank 0 — starts a `wandb.agent` that pulls one hyperparameter
configuration from a sweep and broadcasts it to the other ranks.

**1. Create a W&B sweep** for the run you want (see the swept hyperparameters below), e.g.:

```bash
wandb sweep sweep.yaml        # prints a sweep id like <entity>/<project>/<id>
```

**2. Point the entrypoint at your sweep.** Edit the hardcoded `sweep_id` near the bottom of the
script (and the W&B `entity/project`, currently `wpklab/AIMS`):

```python
# AIMS_fingerprint_dpp.py
wandb.agent(sweep_id="AIMS/mq5r4ud3", count=1, function=...)   # ← change to your sweep id
```

| Entrypoint | Dataset (hardcoded `c`) | Sweep id (hardcoded) |
|------------|-------------------------|----------------------|
| `AIMS_fingerprint_dpp.py` | `aim_all_designs_all_views` | `AIMS/mq5r4ud3` |
| `AIMS_fingerprint_original_model.py` | `aim_all_designs_all_views` | `AIMS/qsds6gvq` |
| `AIMS_fingerprint_design_sweep.py` | `aim_all_designs_all_views_design_separation_20_train_10_val` | `AIMS/oyy4cmab` |

**3. Launch on SLURM** (edit `--account`, `--partition`, node/GPU counts, and the `srun python …`
line in the sbatch to match your entrypoint):

```bash
sbatch aims_H200.sbatch
```

Single-node quick test (no sbatch) — set the env vars `srun` would normally provide:

```bash
SLURM_PROCID=0 WORLD_SIZE=1 SLURM_GPUS_ON_NODE=1 SLURM_JOB_ID=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python AIMS_fingerprint_dpp.py
```

**Outputs** (written under `data/`):
- `data/Models/<dataset>_<wandb_run_id>_dpp.pth` — best checkpoint (saved when val acc > 0.90).
- `data/Results/<dataset>.txt` — loss/accuracy summary.
- `data/<dataset>/csv_outputs/…` — per-epoch predictions vs. ground truth.
- Metrics, confusion matrices, and the sigma schedule are logged to W&B.

### Swept / key hyperparameters

Pulled from `wandb.config` at runtime (define these in your sweep): `lr`, `model_name`,
`weight_decay`, `lr_gamma`, `sigma` (initial region-selection noise), `test_samples` (= number
of patches `k`), `num_epochs`, plus augmentation knobs `sensor_noise`, `contrast_mod`,
`scale_mod`, `clahe_grid`. Transformer geometry (`n_layer`, `n_head`, `d_model=1024`,
`d_inner`, …) and patch size (448) are set as defaults inside the scripts.

## Adapting it to your setup

The scripts are honest research code. To run them on your own data/cluster, expect to touch:

- **Dataset name** — the `c = '<dataset>'` constant near the top of `run_model` in each entrypoint.
- **Class list** — `class_names = [...]` must match your ImageFolder subfolders (and `num_classes`).
- **W&B project/sweep** — replace `sweep_id="AIMS/…"` and the `wpklab/AIMS` entity/project, and
  create the corresponding sweep. A W&B login is required.
- **Data root** — everything is resolved relative to `./data/`; keep that layout or edit `main_dir`/`data_dir`.
- **Cluster specifics** — `--account`, `--partition`, node/GPU counts, and NCCL env vars in
  `aims_H200.sbatch`.
- **Backbone** — select via the `model_name` sweep value; supported names are enumerated in the
  `initialize_model` functions and the entrypoints' model-construction blocks.

## Citation

If you use this code or dataset, please cite the paper (citation to be added):

```bibtex
@article{learned_region_fingerprinting,
  title   = {Learned Region Selection and Deep Learning for Generalized Source
             Identification of Additively Manufactured Parts},
  author  = {WPK Lab},
  year    = {2025}
}
```
