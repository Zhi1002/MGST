import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors




def compute_spatial_adjacency_matrix(spatial_data, k=10, dist_metric='euclidean', method='radius', radius=50):
    if method == 'knn':
        distance_matrix = cdist(spatial_data, spatial_data, metric=dist_metric)
        adj_spot = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity', include_self=False)
    elif method == 'radius':
        if radius is None:
            raise ValueError("Radius must be specified for 'radius' method.")
        nbrs = NearestNeighbors(radius=radius).fit(spatial_data)
        adj_spot = nbrs.radius_neighbors_graph(spatial_data, mode='connectivity')
        adj_spot = adj_spot + sp.eye(adj_spot.shape[0])
    else:
        raise ValueError("Unsupported method. Use 'knn' or 'radius'.")

    return adj_spot

# encode-position
def positional_encoding(coords, output_dim=32):
    assert output_dim % 2 == 0, "output_dim must be even"
    N = coords.shape[0]
    D = output_dim // 2  # half in sin，the half in cos
    theta = np.pi * ((np.arange(D) + 1) / D)
    theta = np.expand_dims(theta, axis=0)
    theta = np.repeat(theta, 2, axis=0)
    sin_enc = np.sin(coords @ theta)
    cos_enc = np.cos(coords @ theta)
    encoding = np.concatenate([sin_enc, cos_enc], axis=-1)
    return encoding

