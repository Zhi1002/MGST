import unittest

import numpy as np
import pandas as pd
import torch

from mgst.graphs import build_edge_features, build_gene_graph, build_image_graph, build_spatial_graph
from mgst.model import GAEVAE
from mgst.preprocessing import augment_gene_data, perform_gene_pca, positional_encoding
from mgst.training import cluster_final_latent, train_model


class SmokeTests(unittest.TestCase):
    def test_graphs_and_edge_descriptors(self):
        generator = np.random.default_rng(42)
        expression = pd.DataFrame(generator.random((20, 16)))
        coordinates = generator.random((16, 2)) * 10
        image_features = generator.random((16, 8))
        gene, gene_mask = build_gene_graph(expression, k=4)
        spatial, spatial_mask = build_spatial_graph(coordinates, radius=30)
        image, image_mask = build_image_graph(image_features, k=5)
        graphs, edges = build_edge_features(
            [gene, spatial, image], [gene_mask, spatial_mask, image_mask]
        )
        self.assertEqual(edges.shape, (16, 16, 9))
        self.assertEqual(len(graphs), 3)
        np.testing.assert_allclose(edges[:, :, [2, 5, 8]],
                                   np.stack([gene_mask.toarray(), spatial_mask.toarray(), image_mask.toarray()], axis=-1))

    def test_preprocessing_dimensions(self):
        generator = np.random.default_rng(7)
        expression = pd.DataFrame(generator.random((30, 16)))
        coordinates = generator.random((16, 2))
        augmented = augment_gene_data(expression.to_numpy(), coordinates, neighbour_k=5)
        pca_components = perform_gene_pca(pd.DataFrame(augmented), n_components=8)
        self.assertEqual(pca_components.shape, (30, 8))
        self.assertEqual(positional_encoding(coordinates).shape, (16, 32))

    def test_final_epoch_training_and_clustering(self):
        torch.manual_seed(42)
        count = 12
        input_dim = 10
        one = torch.eye(count, dtype=torch.float32)
        empty = torch.zeros_like(one)
        edge_descriptors = torch.zeros((count, count, 9), dtype=torch.float32)
        edge_descriptors[:, :, 0] = one
        edge_descriptors[:, :, 1] = 1 - one
        edge_descriptors[:, :, 2] = one
        model = GAEVAE(input_dim, [16, 5], one, empty, empty, edge_descriptors)
        latent = train_model(model, torch.randn(count, input_dim), epochs=2,
                             learning_rate=0.001, beta=0.01, seed=42)
        clusters = cluster_final_latent(latent, n_clusters=3)
        self.assertEqual(latent.shape, (count, 5))
        self.assertEqual(clusters.shape, (count,))


if __name__ == "__main__":
    unittest.main()
