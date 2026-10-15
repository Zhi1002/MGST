import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import normalize


def build_gene_graph(expression, k=10):
    distances = kneighbors_graph(
        expression.T.to_numpy(),
        n_neighbors=k,
        metric="cosine",
        mode="distance",
        include_self=True,
    )
    mask = distances.copy().tocsr()
    mask.data = np.ones_like(mask.data, dtype=np.float32)
    mask = mask.maximum(mask.T)
    adjacency = distances.copy()
    adjacency.data = np.clip(1.0 - adjacency.data, 0.0, 1.0)
    adjacency = (adjacency + adjacency.T) / 2
    return adjacency, mask


def build_spatial_graph(coords, radius=350.0):
    neighbors = NearestNeighbors(radius=radius, metric="euclidean").fit(coords)
    distances = neighbors.radius_neighbors_graph(coords, mode="distance")
    sigma = np.median(distances.data)
    adjacency = distances.copy()
    adjacency.data = np.exp(-(adjacency.data ** 2) / (2 * sigma ** 2))
    adjacency = (adjacency + adjacency.T) / 2
    mask = (adjacency > 0).astype(np.float32).tocsr()
    return adjacency, mask


def build_image_graph(features, k=30):
    features = np.asarray(features, dtype=np.float32)
    n_spots = features.shape[0]
    k = min(k, n_spots - 1)
    features_norm = normalize(features, norm="l2", axis=1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    neighbors.fit(features_norm)
    distances, indices = neighbors.kneighbors(features_norm)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    similarities = np.clip(1.0 - distances, 0.0, 1.0)
    rows = np.repeat(np.arange(n_spots), k)
    cols = indices.reshape(-1)
    weights = similarities.reshape(-1)
    mask = sp.coo_matrix(
        (np.ones_like(weights, dtype=np.float32), (rows, cols)),
        shape=(n_spots, n_spots),
    ).tocsr()
    mask = mask.maximum(mask.T)
    adjacency = sp.coo_matrix(
        (weights, (rows, cols)), shape=(n_spots, n_spots)
    )
    adjacency = (adjacency + adjacency.T) / 2
    return adjacency, mask


def build_edge_features(adjacencies, masks):
    dense_adjacencies = [adj.toarray() for adj in adjacencies]
    dense_masks = [mask.toarray().astype(np.float32) for mask in masks]
    edge_features = np.stack(
        [
            dense_adjacencies[0],
            1 - dense_adjacencies[0],
            dense_masks[0],
            dense_adjacencies[1],
            1 - dense_adjacencies[1],
            dense_masks[1],
            dense_adjacencies[2],
            1 - dense_adjacencies[2],
            dense_masks[2],
        ],
        axis=-1,
    )
    return dense_adjacencies, edge_features
