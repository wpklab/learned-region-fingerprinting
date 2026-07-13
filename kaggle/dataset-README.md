# AIMS: Additive Manufacturing Source Identification (All Designs, All Views)

Images of 3D-printed parts for **source identification** — predicting which printer produced a
part from an image of it. This is the primary dataset for the paper *"Learned Region Selection
and Deep Learning for Generalized Source Identification of Additively Manufactured Parts"*
(code: https://github.com/wpklab/learned-region-fingerprinting).

## Task

6-way classification of the **source printer** — `Stratasys450mc-1 … Stratasys450mc-6` (six
Stratasys 450mc FDM machines). The intended challenge is **generalization**: the same part
geometries and camera viewpoints do not all appear across splits, so a model must key on the
machine's manufacturing "fingerprint" (fine surface texture) rather than part shape.

## Contents

| Split | Images | Type | Notes |
|-------|-------:|------|-------|
| `train` | 178,332 | 2550×2550 PNG renders | training set |
| `val` | 44,616 | 2550×2550 PNG renders | held-out validation |
| `cell_val` | 487 | phone photos (JPG) | real Pixel/iPhone/Samsung captures — sim-to-real test |

Layout (standard `torchvision.ImageFolder`):

```
aim_all_designs_all_views/
  train/Stratasys450mc-1/A22222_azimuth_0_polar_0.png
  train/Stratasys450mc-1/A22222_azimuth_0_polar_14.png
  ...
  val/Stratasys450mc-3/...
  cell_val/Stratasys450mc-5/Pixel_5_Flat_Connector_Wood.jpg
```

## File naming

Rendered images: `{SERIAL}_azimuth_{A}_polar_{P}.png`
- first **6 characters** = the physical part's serial (≈200 distinct parts per printer);
- `azimuth` / `polar` = the camera viewpoint in degrees.

`cell_val` photos are named by `{phone}_{view}_{design}_{background}.jpg`.

## Quick start

```python
from torchvision import datasets, transforms
ds = datasets.ImageFolder("aim_all_designs_all_views/train", transforms.ToTensor())
print(ds.classes)   # ['Stratasys450mc-1', ..., 'Stratasys450mc-6']
```

## License & citation

Released under CC-BY-4.0. If you use this dataset, please cite the paper (see the code repo for
the current citation).
