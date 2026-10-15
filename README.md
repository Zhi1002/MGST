# MGST

Modality-aware graph fusion for spatial transcriptomics. This repository provides the single-dataset training pipeline using gene expression, spatial coordinates, and histology images.

## Requirements

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for full-sized DLPFC slices. The feature extractor uses ImageNet-pretrained ResNet-18 weights, which may be downloaded by torchvision during the first run.

## Input data

By default, the script reads dataset `151671` from a local DLPFC directory. Set the paths in `mgst/config.py`, or override them with environment variables `MGST_DATA_ROOT` and `MGST_OUTPUT_ROOT`.

Expected input layout:

```text
DLPFC/
└── 151671/
    ├── filtered_feature_bc_matrix.h5
    ├── metadata.csv
    └── spatial/
        ├── tissue_hires_image.png
        ├── scalefactors_json.json
        └── tissue_positions_list.csv
```

`tissue_positions.csv` is also supported instead of `tissue_positions_list.csv`. Metadata must contain `barcode`, `imagerow`, and `imagecol`. Optional `layer_guess` labels are used only for a single final ARI calculation, after model training and clustering. No ground-truth labels are used to select an epoch or fit the model.

## Run

```bash
python run.py
```

The console displays a training progress bar without per-epoch loss, ARI, or diagnostic logs. The pipeline saves a single output file:

```text
151671_seed62_clusters.csv
```

The output contains `spot_name` and `cluster`. The clusters come from KMeans applied to the posterior mean representation at the **final epoch**, with no ARI-based epoch selection. No latent embeddings, intermediate epochs, benchmark summaries, or figures are saved.

## Default setup

- Dataset: `151671`; five clusters
- Seed: `62`; 60 epochs
- Learning rate: `0.0005`; beta: `0.01`
- Graph fusion: nine-dimensional edge descriptors and a 9–128–128–3 MLP
- Latent dimension: 64
- Image input: 10x HDF5 gene expression matrix plus the tissue H&E image
- Final clustering: KMeans with `n_init=30`

These settings reproduce the configuration supplied with this code. See `mgst/config.py` for the remaining graph and preprocessing defaults.

## Code layout

- `mgst/config.py`: dataset paths and run settings
- `mgst/preprocessing.py`: HDF5 expression loading and gene/spatial preprocessing
- `mgst/graphs.py`: three modality-specific graphs and edge descriptors
- `mgst/images.py`: pretrained image features and coordinate handling
- `mgst/model.py`: edge-conditioned fusion and variational graph encoder
- `mgst/training.py`: training and final-epoch clustering
- `mgst/pipeline.py`: end-to-end data and output workflow
- `tests/test_smoke.py`: small functional tests
