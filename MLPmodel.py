import torch
import torch.nn as nn






# --------------------------MLP -------------------------- #
class MLPFusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(3, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, A1, A2, A3):
        # A1, A2, A3 shape (N, N)
        A_stack = torch.stack([A1, A2, A3], dim=-1)  # shape: (N, N, 3)
        N = A_stack.shape[0]
        fused_flat = self.mlp(A_stack.view(-1, 3))
        fused = fused_flat.view(N, N)
        fused = torch.sigmoid(fused)
        mask_original = ((A1 > 0) | (A2 > 0) | (A3 > 0)).float()
        fused = fused * mask_original

        # N = fused.size(0)
        # # I=0
        diag_mask = 1 - torch.eye(N, device=fused.device)
        fused = fused * diag_mask
        fused = (fused + fused.t()) / 2
        # Normalized
        row_sum = fused.sum(dim=1, keepdim=True) + 1e-8
        fused = fused / row_sum
        # fused = (fused + fused.t()) / 2
        return fused


