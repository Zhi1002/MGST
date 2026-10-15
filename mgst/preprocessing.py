import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import models, transforms


def preprocess_expression(data):
    data = data[data.astype(bool).mean(axis=1) >= 0.01]
    data = data.loc[data.var(axis=1, numeric_only=True).sort_values().index[-2000:]]
    data = data.fillna(0)
    numeric_columns = data.select_dtypes(include=[float, int])
    if (numeric_columns.sum() <= 0).any():
        raise ValueError("At least one spot has zero total expression after filtering.")
    return numeric_columns.div(numeric_columns.sum())


def augment_gene_data(data, spatial_coords, adj_wt=0.3, neighbour_k=15):
    knn = NearestNeighbors(n_neighbors=neighbour_k, metric="euclidean")
    knn.fit(spatial_coords)
    indices = knn.kneighbors(spatial_coords, return_distance=False)
    adjacent_avg = np.mean(np.take(data, indices, axis=1), axis=2)
    return data + adj_wt * adjacent_avg


def perform_gene_pca(data, n_components):
    numeric_columns = data.select_dtypes(include=[float, int])
    mean = np.mean(numeric_columns, axis=1)
    standardized_data = (numeric_columns.T - mean) / np.std(numeric_columns, axis=1, ddof=1)
    if np.any(np.isinf(standardized_data)) or np.any(np.isnan(standardized_data)):
        standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)
    cov_matrix = np.cov(standardized_data, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    indices = np.argsort(eigenvalues)[::-1][:n_components]
    return np.dot(standardized_data.T, eigenvectors[:, indices])


def positional_encoding(coords, output_dim=32):
    if output_dim % 4 != 0:
        raise ValueError("Positional encoding dimension must be divisible by four.")
    count = output_dim // 4
    theta = np.pi * ((np.arange(count) + 1) / count)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    return np.concatenate(
        [np.sin(x * theta), np.cos(x * theta), np.sin(y * theta), np.cos(y * theta)],
        axis=1,
    )


def load_spot_data(dataset_root):
    expression = pd.read_csv(dataset_root / "filtered_feature_bc_matrix.csv", index_col=0)
    metadata = pd.read_csv(dataset_root / "metadata.csv")
    if metadata["barcode"].duplicated().any() or expression.columns.duplicated().any():
        raise ValueError("Duplicate barcodes are not supported.")
    expression = expression.loc[:, metadata["barcode"]]
    return expression, metadata


def load_hires_coordinates(dataset_root, barcodes):
    with open(dataset_root / "spatial" / "scalefactors_json.json", encoding="utf-8") as f:
        scale = json.load(f)["tissue_hires_scalef"]
    positions_path = dataset_root / "spatial" / "tissue_positions_list.csv"
    if positions_path.exists():
        positions = pd.read_csv(positions_path, header=None, index_col=0).loc[barcodes]
        return positions[[4, 5]].values * scale
    positions = pd.read_csv(dataset_root / "spatial" / "tissue_positions.csv", index_col=0).loc[barcodes]
    return positions[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values * scale


def crop_image(image, coordinates, crop_size=(50, 50)):
    x, y = coordinates
    left = max(x - crop_size[0] // 2, 0)
    upper = max(y - crop_size[1] // 2, 0)
    right = min(x + crop_size[0] // 2, image.width)
    lower = min(y + crop_size[1] // 2, image.height)
    return image.crop((left, upper, right, lower))


def extract_image_features(dataset_root, image_coordinates, device, n_components=100):
    cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1]).to(device)
    cnn.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features = []
    batch_images = []
    with Image.open(dataset_root / "spatial" / "tissue_hires_image.png") as image:
        with torch.no_grad():
            for row_coordinate, col_coordinate in image_coordinates:
                cropped = crop_image(image, (col_coordinate, row_coordinate), crop_size=(50, 50))
                batch_images.append(transform(cropped))
                if len(batch_images) == 64:
                    batch = torch.stack(batch_images).to(device)
                    features.append(cnn(batch).squeeze(-1).squeeze(-1).cpu().numpy())
                    batch_images = []
            if batch_images:
                batch = torch.stack(batch_images).to(device)
                features.append(cnn(batch).squeeze(-1).squeeze(-1).cpu().numpy())
    image_embeddings = np.vstack(features)
    return PCA(n_components=n_components).fit_transform(image_embeddings)


def combine_features(gene_features, spatial_features, image_features):
    gene_norm = StandardScaler().fit_transform(gene_features)
    spatial_norm = StandardScaler().fit_transform(spatial_features)
    image_norm = StandardScaler().fit_transform(image_features)
    return np.hstack((gene_norm, spatial_norm, image_norm))
