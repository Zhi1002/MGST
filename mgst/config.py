import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    dataset_id: str = "151671"
    n_clusters: int = 5
    seed: int = 62
    n_epochs: int = 60
    lr: float = 5e-4
    beta: float = 0.01
    adj_wt: float = 0.3
    neighbor_k: int = 15
    gene_knn: int = 10
    spatial_radius: float = 350.0
    image_knn: int = 30
    gene_pca_components: int = 400
    image_pca_components: int = 100
    image_batch_size: int = 64
    base_root: Path = Path(os.environ.get("MGST_DATA_ROOT", "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC"))
    save_root: Path = Path(os.environ.get("MGST_OUTPUT_ROOT", "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/npj/M6/plot"))
