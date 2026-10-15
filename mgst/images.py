import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms


def image_crop(image, coordinates, crop_size=(50, 50)):
    x, y = coordinates
    left = max(x - crop_size[0] // 2, 0)
    upper = max(y - crop_size[1] // 2, 0)
    right = min(x + crop_size[0] // 2, image.width)
    lower = min(y + crop_size[1] // 2, image.height)
    return image.crop((left, upper, right, lower))


def load_image_coordinates(data_root, barcodes):
    spatial_root = Path(data_root) / "spatial"
    with (spatial_root / "scalefactors_json.json").open(encoding="utf-8") as handle:
        scale_factors = json.load(handle)
    positions_path = spatial_root / "tissue_positions_list.csv"
    if positions_path.exists():
        positions = pd.read_csv(positions_path, header=None, index_col=0).loc[barcodes]
        coords = positions[[4, 5]].values
    else:
        positions = pd.read_csv(spatial_root / "tissue_positions.csv", index_col=0).loc[barcodes]
        coords = positions[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    return coords * scale_factors["tissue_hires_scalef"]


def extract_image_features(data_root, barcodes, device, batch_size=64, n_components=100):
    data_root = Path(data_root)
    image_coordinates = load_image_coordinates(data_root, barcodes)
    weights = models.ResNet18_Weights.DEFAULT
    cnn_model = models.resnet18(weights=weights)
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1]).to(device)
    cnn_model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    features = []
    batch = []
    with Image.open(data_root / "spatial" / "tissue_hires_image.png") as image:
        with torch.no_grad():
            for index in range(len(barcodes)):
                row_coordinate, col_coordinate = image_coordinates[index]
                crop = image_crop(
                    image,
                    (col_coordinate, row_coordinate),
                    crop_size=(50, 50),
                )
                batch.append(transform(crop))
                if len(batch) == batch_size:
                    features.append(
                        cnn_model(torch.stack(batch).to(device))
                        .squeeze(-1)
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                    batch = []
            if batch:
                features.append(
                    cnn_model(torch.stack(batch).to(device))
                    .squeeze(-1)
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
    extracted = np.vstack(features)
    return PCA(n_components=n_components).fit_transform(extracted)
