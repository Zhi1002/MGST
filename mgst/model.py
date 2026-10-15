import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPFusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, edge_features, gene_adjacency, spatial_adjacency, image_adjacency):
        n_spots = gene_adjacency.shape[0]
        flat = edge_features.reshape(-1, 9)
        modality_mask = flat[:, [2, 5, 8]] > 0.5
        multi_mask = modality_mask.sum(dim=1) > 1
        multi_indices = torch.where(multi_mask)[0]
        logits = self.mlp(flat[multi_mask])
        logits = logits.masked_fill(~modality_mask[multi_mask], -1e9)
        multi_weights = F.softmax(logits, dim=1)
        multi_weights = multi_weights * modality_mask[multi_mask].to(logits.dtype)
        multi_weights = multi_weights / (multi_weights.sum(dim=1, keepdim=True) + 1e-12)
        weights = modality_mask.to(flat.dtype)
        weights = weights.index_copy(0, multi_indices, multi_weights)
        weights = weights.reshape(n_spots, n_spots, 3)
        adjacency_stack = torch.stack(
            [gene_adjacency, spatial_adjacency, image_adjacency], dim=-1
        )
        fused = torch.sum(weights * adjacency_stack, dim=-1)
        fused = (fused + fused.t()) / 2
        fused = fused + torch.eye(n_spots, device=fused.device, dtype=fused.dtype)
        degree = fused.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        return degree_inv_sqrt[:, None] * fused * degree_inv_sqrt[None, :]


class GAEVAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, adjacencies, edge_features):
        super().__init__()
        self.fusion = MLPFusion(hidden_dim=128)
        self.register_buffer("A1", adjacencies[0])
        self.register_buffer("A2", adjacencies[1])
        self.register_buffer("A3", adjacencies[2])
        self.register_buffer("edge_features", edge_features)
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc_mu = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_logvar = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], in_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(
            std.shape,
            device=std.device,
            dtype=std.dtype,
            generator=getattr(self, "training_generator", None),
        )
        return mu + eps * std

    def forward(self, x):
        adjacency = self.fusion(
            self.edge_features,
            self.A1,
            self.A2,
            self.A3,
        )
        hidden = F.relu(self.fc1(torch.mm(adjacency, x)))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decoder(z), mu, logvar, adjacency
