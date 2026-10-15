import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import normalize


def build_gene_graph(processed_expression, k=10):
    distances = kneighbors_graph(
        processed_expression.T,
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


def build_spatial_graph(spatial_data, radius=350):
    neighbors = NearestNeighbors(radius=radius, metric="euclidean").fit(spatial_data)
    distances = neighbors.radius_neighbors_graph(spatial_data, mode="distance")
    sigma = np.median(distances.data)
    adjacency = distances.copy()
    adjacency.data = np.exp(-(adjacency.data ** 2) / (2 * sigma ** 2))
    adjacency = (adjacency + adjacency.T) / 2
    mask = (adjacency > 0).astype(np.float32).tocsr()
    return adjacency, mask


def build_image_graph(features, k=30):
    features = np.asarray(features, dtype=np.float32)
    count = features.shape[0]
    k = min(k, count - 1)
    normalized = normalize(features, norm="l2", axis=1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    neighbors.fit(normalized)
    distances, indices = neighbors.kneighbors(normalized)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    similarities = np.clip(1.0 - distances, 0.0, 1.0)
    rows = np.repeat(np.arange(count), k)
    columns = indices.reshape(-1)
    values = similarities.reshape(-1)
    mask = sp.coo_matrix(
        (np.ones_like(values, dtype=np.float32), (rows, columns)),
        shape=(count, count),
    ).tocsr()
    mask = mask.maximum(mask.T)
    adjacency = sp.coo_matrix((values, (rows, columns)), shape=(count, count))
    adjacency = (adjacency + adjacency.T) / 2
    return adjacency, mask


def build_edge_features(adjacencies, masks):
    matrices = [matrix.toarray() for matrix in adjacencies]
    presence = [mask.toarray().astype(np.float32) for mask in masks]
    edge_features = np.stack(
        [
            matrices[0], 1 - matrices[0], presence[0],
            matrices[1], 1 - matrices[1], presence[1],
            matrices[2], 1 - matrices[2], presence[2],
        ],
        axis=-1,
    )
    return matrices, edge_features
