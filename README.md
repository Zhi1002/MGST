# MGST

A modular implementation of MGST with 9-dimensional edge descriptors, edge-conditioned multimodal graph fusion, and a VAE-based graph encoder.

## Setup

```bash
python -m pip install -r requirements.txt
```

Configure the input and output directories in `mgst/config.py`. The defaults retain the paths and active dataset from the supplied script.

The expected input files are:

```text
DLPFC/
└── 151671/
    ├── filtered_feature_bc_matrix.csv
    ├── metadata.csv
    └── spatial/
        ├── tissue_hires_image.png
        ├── scalefactors_json.json
        └── tissue_positions_list.csv
```

`tissue_positions.csv` is also supported in place of `tissue_positions_list.csv`.

## Run

```bash
python run.py
```

The only normal console output is the training progress bar. The only generated output file is:

```text
<OUTPUT_ROOT>/151671_seed42_clusters.csv
```

The CSV contains `spot_name` and `cluster`. It stores the final-epoch KMeans clustering, not the epoch with the highest ARI. Reference labels are not used for training, selection, or clustering; if present, they are accessed once after training for a final-epoch ARI calculation that is neither printed nor saved.

The feature construction, spatial radius, graph similarity calculations, 9D edge descriptors, fusion MLP (9 → 128 → 128 → 3), VAE encoder/decoder, loss schedule, and hyperparameters follow the active code path of the supplied script. No plots, intermediate latents, per-epoch results, summaries, or training logs are generated.

The first execution may require downloading the pretrained ResNet-18 weights unless they are already cached by PyTorch. The dataset and pretrained weights are not bundled in this repository.

## Layout

```text
run.py
mgst/
  config.py
  preprocessing.py
  graphs.py
  model.py
  training.py
  pipeline.py
requirements.txt
```
