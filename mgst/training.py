import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.auto import tqdm


def train_model(model, features, n_epochs=60, lr=5e-4, beta=0.01, seed=62, n_clusters=5):
    device = next(model.parameters()).device
    features = features.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.training_generator = torch.Generator(device=device).manual_seed(seed + 100003)

    for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch", dynamic_ncols=True):
        model.train()
        optimizer.zero_grad()
        reconstruction, mu, logvar, _ = model(features)
        reconstruction_loss = F.mse_loss(reconstruction, features)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        current_beta = beta * min(1.0, float(epoch + 1) / 50.0)
        loss = reconstruction_loss + current_beta * kl_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered.")
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                param.grad.detach().square().sum()
                for param in model.parameters()
                if param.grad is not None
            )
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradients encountered.")
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, mu, _, _ = model(features)
    latent = mu.detach().cpu().numpy().astype("float32", copy=True)
    predicted_labels = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=seed,
        n_init=30,
    ).fit_predict(latent)
    return predicted_labels
