import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from . import config
from .graphs import build_edge_features, build_gene_graph, build_image_graph, build_spatial_graph
from .model import GAEVAE
from .preprocessing import (
    augment_gene_data,
    combine_features,
    extract_image_features,
    load_hires_coordinates,
    load_spot_data,
    perform_gene_pca,
    positional_encoding,
    preprocess_expression,
)
from .training import cluster_final_latent, train_model


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_last_epoch(labels, metadata):
    if "layer_guess" not in metadata:
        return float("nan")
    valid = metadata["layer_guess"].notna().to_numpy()
    if valid.sum() < 2:
        return float("nan")
    reference = LabelEncoder().fit_transform(metadata.loc[valid, "layer_guess"])
    return adjusted_rand_score(reference, labels[valid])


def run():
    torch.set_num_threads(4)
    set_global_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = config.DATA_ROOT / config.DATASET_ID

    expression, metadata = load_spot_data(dataset_root)
    processed = preprocess_expression(expression)
    spatial_data = metadata[["imagerow", "imagecol"]].values
    augmented = augment_gene_data(
        processed.values,
        spatial_data,
        adj_wt=config.ADJ_WT,
        neighbour_k=config.GENE_AUGMENT_NEIGHBORS,
    )
    processed = pd.DataFrame(augmented, index=processed.index, columns=processed.columns)
    gene_components = perform_gene_pca(processed, config.GENE_COMPONENTS)
    gene_features = processed.T.dot(gene_components).astype(float)

    a_gene, m_gene = build_gene_graph(processed, k=config.GENE_NEIGHBORS)
    a_spatial, m_spatial = build_spatial_graph(spatial_data, radius=config.SPATIAL_RADIUS)
    spatial_features = positional_encoding(spatial_data, output_dim=32)

    image_coordinates = load_hires_coordinates(dataset_root, metadata["barcode"])
    image_features = extract_image_features(
        dataset_root, image_coordinates, device, n_components=config.IMAGE_COMPONENTS
    )
    a_image, m_image = build_image_graph(image_features, k=config.IMAGE_NEIGHBORS)

    all_features = combine_features(gene_features, spatial_features, image_features)
    adjacencies, edge_features = build_edge_features(
        [a_gene, a_spatial, a_image], [m_gene, m_spatial, m_image]
    )

    features = torch.tensor(all_features, device=device, dtype=torch.float32)
    graphs = [torch.tensor(a, device=device, dtype=torch.float32) for a in adjacencies]
    edges = torch.tensor(edge_features, device=device, dtype=torch.float32)

    set_global_seed(config.SEED)
    model = GAEVAE(
        in_dim=all_features.shape[1],
        hidden_dims=[256, config.LATENT_DIM],
        a_gene=graphs[0],
        a_spatial=graphs[1],
        a_image=graphs[2],
        edge_features=edges,
    ).to(device)

    latent = train_model(
        model,
        features,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        beta=config.BETA,
        seed=config.SEED,
    )
    labels = cluster_final_latent(latent, n_clusters=config.N_CLUSTERS, seed=config.SEED)
    _final_ari = evaluate_last_epoch(labels, metadata)

    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_file = config.OUTPUT_ROOT / f"{config.DATASET_ID}_seed{config.SEED}_clusters.csv"
    pd.DataFrame({"spot_name": metadata["barcode"].values, "cluster": labels}).to_csv(
        output_file, index=False
    )
