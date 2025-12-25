import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from MLPmodel import MLPFusion




class GAE_VAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, A1, A2, A3):
        super().__init__()
        self.fusion = MLPFusion(hidden_dim=8)
        self.register_buffer('A1', A1)
        self.register_buffer('A2', A2)
        self.register_buffer('A3', A3)

        # encode
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc_mu = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_logvar = nn.Linear(hidden_dims[0], hidden_dims[1])

        # Decode
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], in_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        A_fused = self.fusion(self.A1, self.A2, self.A3)
        N = A_fused.size(0)
        A_fused = A_fused * (1 - torch.eye(N, device=A_fused.device))
        row_sum = A_fused.sum(dim=1, keepdim=True) + 1e-8
        A_normalized = A_fused / row_sum

        # h
        h1 = torch.mm(A_normalized, x)
        h1 = F.relu(self.fc1(h1))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decoder(z)
        return x_recon, mu, logvar, A_normalized
    # def forward(self, x):
    #     A_fused = self.fusion(self.A1, self.A2, self.A3)
    #     N = A_fused.size(0)
    #     A_fused = A_fused * (1 - torch.eye(N, device=A_fused.device))
    #     # row_sum = A_fused.sum(dim=1, keepdim=True) + 1e-8
    #     # A_normalized = A_fused / row_sum
    #     # ===== normalized begin =====
    #     deg = A_fused.sum(1)  # degree vector (N,)
    #     deg_inv_sqrt = deg.pow(-0.5)  # D^{-0.5}
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.  # avoid division by zero
    #     D_inv_sqrt = torch.diag(deg_inv_sqrt)  # construct diagonal matrix
    #     A_normalized = D_inv_sqrt @ A_fused @ D_inv_sqrt  # D^{-0.5} A D^{-0.5}
    #
    #     # extract information via graph convolution (adjacency matrix multiplied by input)
    #     h1 = torch.mm(A_normalized, x)
    #     h1 = F.relu(self.fc1(h1))
    #     mu = self.fc_mu(h1)
    #     logvar = self.fc_logvar(h1)
    #     z = self.reparameterize(mu, logvar)
    #
    #     x_recon = self.decoder(z)
    #     return x_recon, mu, logvar, A_normalized

# ------------------------ Training part ------------------------#
def train_model(model, X, true_labels, n_epochs=200, lr=0.01, beta=0.01, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_ari = -1
    best_weights = None
    best_latent = None
    best_labels = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        x_recon, mu, logvar, A = model(X)
        # reconstruction loss
        recon_loss = F.mse_loss(x_recon, X)
        # KL divergence loss (note: summed over all samples, can divide by sample count to balance values)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon_loss + beta * kl_loss

        loss.backward()
        optimizer.step()

        # Evaluate clustering performance
        with torch.no_grad():
            latent = mu.cpu().detach().numpy()  # use mu or z as embedding
            # kmeans = KMeans(n_clusters=7).fit(latent)
            kmeans = KMeans(n_clusters=7, init='k-means++').fit(latent)
            ari = adjusted_rand_score(true_labels, kmeans.labels_)

            if ari > best_ari:
                best_ari = ari
                best_weights = model.state_dict().copy()
                best_latent = latent
                best_labels = kmeans.labels_

        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs} | Loss: {loss.item():.4f} | Recon Loss: {recon_loss.item():.4f} | KL Loss: {kl_loss.item():.4f} | ARI: {ari:.4f}')

    print(f"Training finished. Best ARI: {best_ari:.4f}")
    model.load_state_dict(best_weights)
    return model, best_latent, best_labels


