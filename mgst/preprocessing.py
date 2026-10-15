import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def read_expression_h5(filename):
    with h5py.File(filename, "r") as handle:
        matrix = handle["matrix"]
        counts = sp.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=tuple(matrix["shape"][:]),
        )
        features = matrix["features"]
        gene_names = features["name"].asstr()[:]
        barcodes = matrix["barcodes"].asstr()[:]
        if "feature_type" in features:
            keep = features["feature_type"].asstr()[:] == "Gene Expression"
            counts = counts[keep, :]
            gene_names = gene_names[keep]
    dtype = np.int64 if np.issubdtype(counts.dtype, np.integer) else np.float64
    return pd.DataFrame(
        counts.astype(dtype, copy=False).toarray(),
        index=gene_names,
        columns=barcodes,
    )


def preprocess_expression(data):
    data = data[data.astype(bool).mean(axis=1) >= 0.01]
    data = data.loc[data.var(axis=1, numeric_only=True).sort_values().index[-2000:]].copy()
    data.fillna(0, inplace=True)
    numeric_columns = data.select_dtypes(include=[float, int])
    if (numeric_columns.sum() <= 0).any():
        raise ValueError("An expression column has zero total counts after gene filtering.")
    return numeric_columns.div(numeric_columns.sum())


def augment_gene_data(data, spatial_coords, adj_wt, neighbor_k=15):
    knn = NearestNeighbors(n_neighbors=neighbor_k, metric="euclidean")
    knn.fit(spatial_coords)
    indices = knn.kneighbors(spatial_coords, return_distance=False)
    adjacent_avg = np.mean(np.take(data, indices, axis=1), axis=2)
    return data + adj_wt * adjacent_avg


def perform_pca(data, n_components=400):
    numeric_columns = data.select_dtypes(include=[float, int])
    mean = np.mean(numeric_columns, axis=1)
    standardized = (numeric_columns.T - mean) / np.std(numeric_columns, axis=1, ddof=1)
    if np.any(np.isinf(standardized)) or np.any(np.isnan(standardized)):
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = np.cov(standardized, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    top_indices = np.argsort(eigenvalues)[::-1][:n_components]
    return np.dot(standardized.T, eigenvectors[:, top_indices])


def positional_encoding(coords, output_dim=32):
    if output_dim % 4 != 0:
        raise ValueError("Positional encoding dimension must be divisible by four.")
    dimensions = output_dim // 4
    theta = np.pi * ((np.arange(dimensions) + 1) / dimensions)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    return np.concatenate(
        [np.sin(x * theta), np.cos(x * theta), np.sin(y * theta), np.cos(y * theta)],
        axis=1,
    )
