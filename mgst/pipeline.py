import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config
from .graphs import build_edge_features, build_gene_graph, build_image_graph, build_spatial_graph
from .images import extract_image_features
from .model import GAEVAE
from .preprocessing import augment_gene_data, perform_pca, positional_encoding, preprocess_expression, read_expression_h5
from .training import train_model


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_final_ari(metadata, predicted_labels):
    if "layer_guess" not in metadata.columns:
        return None
    valid = metadata["layer_guess"].notna().to_numpy()
    if valid.sum() < 2:
        return None
    ground_truth = LabelEncoder().fit_transform(
        metadata.loc[valid, "layer_guess"]
    ) + 1
    return adjusted_rand_score(ground_truth, predicted_labels[valid])


def run(config=None):
    config = config or Config()
    set_global_seed(config.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = config.base_root / config.dataset_id
    expression = read_expression_h5(data_root / "filtered_feature_bc_matrix.h5")
    metadata = pd.read_csv(data_root / "metadata.csv")
    if metadata["barcode"].duplicated().any() or expression.columns.duplicated().any():
        raise ValueError("Duplicate barcodes detected.")
    expression = expression.loc[:, metadata["barcode"]]
    processed = preprocess_expression(expression)
    spatial_coords = metadata[["imagerow", "imagecol"]].values
    augmented = augment_gene_data(
        processed.values,
        spatial_coords,
        config.adj_wt,
        neighbor_k=config.neighbor_k,
    )
    processed = pd.DataFrame(augmented, index=processed.index, columns=processed.columns)
    gene_pca = perform_pca(processed, n_components=config.gene_pca_components)
    gene_adjacency, gene_mask = build_gene_graph(processed, k=config.gene_knn)
    gene_features = pd.DataFrame(processed.T.dot(gene_pca).astype(float))
    spatial_adjacency, spatial_mask = build_spatial_graph(
        spatial_coords, radius=config.spatial_radius
    )
    spatial_features = positional_encoding(spatial_coords, output_dim=32)
    image_features = extract_image_features(
        data_root,
        metadata["barcode"],
        device,
        batch_size=config.image_batch_size,
        n_components=config.image_pca_components,
    )
    image_adjacency, image_mask = build_image_graph(image_features, k=config.image_knn)
    feature_blocks = [
        StandardScaler().fit_transform(gene_features),
        StandardScaler().fit_transform(spatial_features),
        StandardScaler().fit_transform(image_features),
    ]
    combined = np.hstack(feature_blocks)
    features = torch.tensor(combined, dtype=torch.float32, device=device)
    adjacencies, edge_features = build_edge_features(
        [gene_adjacency, spatial_adjacency, image_adjacency],
        [gene_mask, spatial_mask, image_mask],
    )
    adjacencies = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adjacencies]
    edge_features = torch.tensor(edge_features, dtype=torch.float32, device=device)
    set_global_seed(config.seed)
    model = GAEVAE(
        in_dim=combined.shape[1],
        hidden_dims=[256, 64],
        adjacencies=adjacencies,
        edge_features=edge_features,
    ).to(device)
    predicted_labels = train_model(
        model,
        features,
        n_epochs=config.n_epochs,
        lr=config.lr,
        beta=config.beta,
        seed=config.seed,
        n_clusters=config.n_clusters,
    )
    output_dir = config.save_root
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"spot_name": metadata["barcode"].to_numpy(), "cluster": predicted_labels}
    ).to_csv(
        output_dir / f"{config.dataset_id}_seed{config.seed}_clusters.csv",
        index=False,
    )
    evaluate_final_ari(metadata, predicted_labels)
