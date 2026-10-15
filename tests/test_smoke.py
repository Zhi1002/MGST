import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from mgst.graphs import build_edge_features, build_gene_graph, build_image_graph
from mgst.model import GAEVAE
from mgst.pipeline import evaluate_final_ari
from mgst.preprocessing import augment_gene_data, positional_encoding
from mgst.training import train_model


def test_spatial_encoding_and_augmentation():
    coords = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    enc = positional_encoding(coords, output_dim=32)
    assert enc.shape == (3, 32)
    data = np.ones((5, 3))
    result = augment_gene_data(data, coords, 0.3, neighbor_k=2)
    assert result.shape == data.shape
    np.testing.assert_allclose(result, 1.3)


def test_nine_dimensional_edges():
    expression = pd.DataFrame(np.array([[1., 2., 3., 4.], [5., 4., 3., 2.]]))
    gene_adj, gene_mask = build_gene_graph(expression, k=2)
    image_adj, image_mask = build_image_graph(np.arange(20, dtype=float).reshape(4, 5), k=2)
    spatial_adj = sp.eye(4, format="csr")
    spatial_mask = sp.eye(4, format="csr")
    dense_adj, features = build_edge_features(
        [gene_adj, spatial_adj, image_adj],
        [gene_mask, spatial_mask, image_mask],
    )
    assert len(dense_adj) == 3
    assert features.shape == (4, 4, 9)


def test_final_epoch_training_and_final_only_evaluation():
    torch.manual_seed(62)
    n_spots = 4
    adjacency = torch.eye(n_spots)
    edge_features = torch.zeros(n_spots, n_spots, 9)
    edge_features[:, :, 0] = adjacency
    edge_features[:, :, 1] = 1 - adjacency
    edge_features[:, :, 2] = adjacency
    model = GAEVAE(6, [8, 3], [adjacency] * 3, edge_features)
    labels = train_model(model, torch.rand(n_spots, 6), n_epochs=2, seed=62, n_clusters=3)
    assert labels.shape == (n_spots,)
    assert len(set(labels.tolist())) <= 5
    metadata = pd.DataFrame({"layer_guess": ["A", "B", None, "A"]})
    assert isinstance(evaluate_final_ari(metadata, labels), float)


def test_h5_reader_and_single_output_pipeline(tmp_path, monkeypatch):
    import h5py
    import mgst.pipeline as pipeline
    from mgst.config import Config
    from mgst.preprocessing import read_expression_h5

    barcodes = [f"spot_{index}" for index in range(7)]
    dataset = tmp_path / "input" / "151671"
    dataset.mkdir(parents=True)
    counts = np.arange(1, 57, dtype=np.int64).reshape(8, 7)
    csc = sp.csc_matrix(counts)
    with h5py.File(dataset / "filtered_feature_bc_matrix.h5", "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("data", data=csc.data)
        matrix.create_dataset("indices", data=csc.indices)
        matrix.create_dataset("indptr", data=csc.indptr)
        matrix.create_dataset("shape", data=csc.shape)
        features = matrix.create_group("features")
        strings = h5py.string_dtype(encoding="utf-8")
        features.create_dataset("name", data=[f"g{i}" for i in range(8)], dtype=strings)
        features.create_dataset("feature_type", data=["Gene Expression"] * 8, dtype=strings)
        matrix.create_dataset("barcodes", data=barcodes, dtype=strings)
    metadata = pd.DataFrame({
        "barcode": barcodes,
        "imagerow": np.arange(7, dtype=float) * 20,
        "imagecol": np.arange(7, dtype=float) * 25,
        "layer_guess": ["L1", "L2", "L1", "L2", "L1", "L2", None],
    })
    metadata.to_csv(dataset / "metadata.csv", index=False)
    expression = read_expression_h5(dataset / "filtered_feature_bc_matrix.h5")
    np.testing.assert_array_equal(expression.values, counts)

    monkeypatch.setattr(
        pipeline,
        "extract_image_features",
        lambda data_root, barcodes, device, batch_size, n_components:
            np.arange(len(barcodes) * n_components, dtype=float).reshape(len(barcodes), n_components),
    )
    config = Config(
        base_root=tmp_path / "input",
        save_root=tmp_path / "output",
        n_epochs=2,
        n_clusters=3,
        gene_knn=3,
        neighbor_k=3,
        image_knn=3,
        gene_pca_components=4,
        image_pca_components=4,
    )
    pipeline.run(config)
    saved_files = list(config.save_root.rglob("*"))
    assert len(saved_files) == 1
    assert saved_files[0].name == "151671_seed62_clusters.csv"
    clusters = pd.read_csv(saved_files[0])
    assert clusters.columns.tolist() == ["spot_name", "cluster"]
    assert clusters["spot_name"].tolist() == barcodes
    assert len(clusters) == 7
