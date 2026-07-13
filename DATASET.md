# Dataset guide

All datasets follow a `torchvision.ImageFolder` layout under `./data/<dataset_name>/{train,val}/`,
with one subfolder per **source class**. Images are 2550×2550 RGB PNG renders named
`{SERIAL}_azimuth_{A}_polar_{P}.png` (first 6 chars = physical-part serial, remainder = camera
viewpoint). Average file size ≈ **0.62 MB**.

## Important: most datasets share the same images

The many `aim_*` folders are **re-partitions of a single underlying render pool** (the same
part serials and viewpoint images), reorganized to test different generalization axes:

| Dataset | Layout / purpose | Approx. images | Approx. size* |
|---------|------------------|---------------:|--------------:|
| `aim_all_designs_all_views` | **Canonical pool**, flat per printer. Used by the main DPS entrypoint and the baseline. | ~223,000 | 141 GB |
| `aim_all_designs_all_views_design_separation_20_train_10_val` | Same images, nested by design; train/val split by **held-out part geometry**. | ~213,000 | ~130 GB |
| `aim_all_designs_all_views_azimuth_separation_180_train_90_val` | Same images, split by **camera azimuth** (viewpoint generalization). | ~213,000 | ~130 GB |
| `aim_all_designs_planar_view_180_train_90_val` | Planar-view subset. | ~77,000 | ~47 GB |
| `aim_3_designs_all_views_cell` | 3-design subset incl. cell-phone-captured test images. | ~71,000 | ~43 GB |
| `printer_*`, `carbon_*`, `material_*`, `build*_*` | Smaller cross-process / material / build-location experiments (DLS, SLS, MJF, FDM, MEGA, Carbon). | 100s – few 1000s each | MBs – few GB each |
| `fast_test_*` | Tiny subsets for smoke-testing the pipeline. | small | small |

\* Estimated as `image_count × 0.62 MB`; not a precise `du`. The full `data/` tree
(all re-partitions + `Models/`, `Results/`, `csv_outputs/`) is well over 500 GB, but the
**unique image content is ~141 GB** (the canonical pool) plus the small cross-process sets.

## What to publish on Kaggle (recommended)

Uploading every folder would push ~400+ GB of **duplicated pixels** (the `design_separation`,
`azimuth_separation`, and `planar_view` sets are the canonical images re-split). Instead:

**Tier 1 — reproduce the headline result (recommended minimum).**
Upload **`aim_all_designs_all_views`** (141 GB). Both the DPS method
(`AIMS_fingerprint_dpp.py`) and the baseline (`AIMS_fingerprint_original_model.py`) train on it,
so this alone reproduces the core source-identification result.

**Tier 2 — reproduce the generalization experiments, without duplication.**
The design/azimuth/planar splits are derived from Tier 1. Rather than re-upload ~260 GB of the
same pixels, publish **lightweight split manifests** (a CSV per split listing
`relative_path, class, split`) plus a small `build_split.py` that recreates each
`ImageFolder` split from the canonical pool via symlinks. Ship the manifests in the repo; they
are a few MB. (If symlink-free `ImageFolder` folders are strictly required, upload the splits as
separate Kaggle datasets — but flag the duplication.)

**Tier 3 — the broader cross-process study.**
Bundle the small `printer_*`, `carbon_*`, `material_*`, and `build*_*` folders into one modest
Kaggle dataset (a few GB total). These are cheap and make the process/material/build-location
experiments reproducible.

**Skip:** `fast_test_*` (smoke tests), `Models/`, `Results/`, `csv_outputs/`, `image_outputs/`,
and the `wandb/` run directory — these are derived artifacts, not source data.

### Practical Kaggle notes
- 141 GB is large for a single Kaggle dataset. Options: split the canonical pool into 2–3
  Kaggle dataset "versions"/parts (e.g. by printer class), or upload via the Kaggle **API**
  (`kaggle datasets create`), which handles large uploads better than the web UI.
- These are renders with fine surface texture that carries the "fingerprint" signal. Prefer
  **lossless PNG** as shipped. If size forces compression, publish a clearly-labeled
  high-quality JPEG variant *in addition to* (not instead of) a lossless reference subset, and
  note that lossy compression may attenuate the fine-texture signal the model relies on.
- Include a short `README` on the Kaggle dataset page documenting the class list, the
  `{SERIAL}_azimuth_{A}_polar_{P}.png` naming, and the expected `data/<name>/{train,val}/<class>/`
  layout so users can drop it straight under `./data/`.
