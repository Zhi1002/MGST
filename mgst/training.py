import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.auto import tqdm


def train_model(model, features, epochs, learning_rate, beta, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.training_generator = torch.Generator(device=features.device).manual_seed(seed + 100003)

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch", dynamic_ncols=True):
        model.train()
        optimizer.zero_grad()
        reconstruction, mu, logvar, _ = model(features)
        reconstruction_loss = F.mse_loss(reconstruction, features)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        current_beta = beta * min(1.0, float(epoch + 1) / 30.0)
        loss = reconstruction_loss + current_beta * kl_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}.")
        loss.backward()
        grad_norm = torch.sqrt(
            sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradients at epoch {epoch + 1}.")
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, mu, _, _ = model(features)
    return mu.detach().cpu().numpy().astype(np.float32, copy=True)


def cluster_final_latent(latent, n_clusters, seed=42):
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=seed,
        n_init=10,
    ).fit_predict(latent)
