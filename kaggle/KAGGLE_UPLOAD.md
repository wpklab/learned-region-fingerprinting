# Uploading the Tier-1 dataset to Kaggle

This publishes **`aim_all_designs_all_views`** — the canonical render pool used by both the DPS
method (`AIMS_fingerprint_dpp.py`) and the baseline (`AIMS_fingerprint_original_model.py`).

## What gets uploaded

```
aim_all_designs_all_views/
├── train/       178,332 images  (2550×2550 PNG renders, 6 printer classes)
├── val/          44,616 images  (2550×2550 PNG renders, 6 printer classes)
└── cell_val/        487 images  (real Pixel/iPhone/Samsung photos, .jpg — real-world test set)
```

- **Classes:** `Stratasys450mc-1 … Stratasys450mc-6` (present in every split).
- **Total:** ~223,435 images, **141 GB** on disk (train 112 GB, val 29 GB, cell_val 771 MB).
- `train/` and `val/` are synthetic renders (PNG); `cell_val/` is real phone photos (JPG) for
  sim-to-real evaluation. All three are worth keeping.

## Prerequisites (on a machine with the data + internet)

```bash
pip install kaggle
# Create an API token at https://www.kaggle.com/settings -> "Create New Token"
# It downloads kaggle.json; then:
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

## Step 1 — put the metadata file next to the data

The Kaggle CLI uploads every file in the folder you point `-p` at, and expects a
`dataset-metadata.json` in that folder. Copy the template from this repo and **edit the `id`**
to `<your-kaggle-username>/aims-am-source-identification`:

```bash
cp kaggle/dataset-metadata.json /path/to/data/aim_all_designs_all_views/dataset-metadata.json
# then edit the "id" field: replace INSERT_KAGGLE_USERNAME with your username
```

## Step 2 — create the dataset

```bash
kaggle datasets create \
  -p /path/to/data/aim_all_designs_all_views \
  --dir-mode zip
```

Notes:
- `--dir-mode zip` packs the directory tree into archives before upload — strongly recommended
  here (223k small files upload far faster zipped than individually).
- This is a **~141 GB upload**; run it from a wired/stable connection and expect it to take a
  while. Use `tmux`/`screen` or `nohup` so it survives a disconnect.
- To publish updates later: `kaggle datasets version -p <folder> -m "message" --dir-mode zip`.

## Step 3 — finish on the website

The dataset is created **private**. On its Kaggle page:
- Add the description (see `kaggle/dataset-README.md`), confirm the license, add tags
  (e.g. *additive manufacturing, computer vision, image classification, materials*).
- Set visibility to **Public** when ready.

## Step 4 — link it back to the code

Once public, update the code repo's `README.md` / `DATASET.md` "Dataset download" line with the
Kaggle URL, and (optionally) the `data/` setup step:

```bash
kaggle datasets download -d <your-username>/aims-am-source-identification -p data/ --unzip
# result: data/aim_all_designs_all_views/{train,val,cell_val}/...
```

## Size caveat

If the upload stalls on Kaggle's per-dataset limits, split into parts (e.g. one dataset per
printer class, or train / val+cell_val separately) and note the split on each dataset page.
Keep the images as **lossless PNG** — the source-identification signal lives in fine surface
texture that lossy recompression can wash out.
