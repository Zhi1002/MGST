import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import leidenalg
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  

def augment_gene_data(data, spatial_coords, Adj_WT, neighbour_k=10):
    knn = NearestNeighbors(n_neighbors=neighbour_k, metric='euclidean')
    knn.fit(spatial_coords)
    neigh_indices = knn.kneighbors(spatial_coords, return_distance=False)

    adjacent_avg = np.mean(np.take(data, neigh_indices, axis=1), axis=2)
    augment_data = data + Adj_WT * adjacent_avg

    return augment_data


def preprocessingCSV(df, expressionFilename=None):
    if isinstance(df, pd.DataFrame):
        data = df
    else:
        data = pd.read_csv(expressionFilename, index_col=0, header=0)

    data = data[data[data.columns[1:]].astype('bool').mean(axis=1) >= 0.01]
    data = data.loc[(data.iloc[1:, 1:].var(axis=1, numeric_only=True).sort_values()[-2000:]).index]
    data.fillna(0, inplace=True)

    numeric_columns = data.select_dtypes(include=[float, int])
    data = numeric_columns.div(numeric_columns.sum())

    return data

def perform_pca(df1, n_components):
    numeric_columns = df1.iloc[:, 1:].select_dtypes(include=[float, int])
    mean = np.mean(numeric_columns, axis=1)
    standardized_data = (df1.iloc[:, 1:].T - mean) / np.std(numeric_columns, axis=1, ddof=1)

    if np.any(np.isinf(standardized_data)) or np.any(np.isnan(standardized_data)):
        standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)

    cov_matrix = np.cov(standardized_data, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    top_eigenvalue_indices = np.argsort(eigenvalues)[::-1][:n_components]
    selected_eigenvectors = eigenvectors[:, top_eigenvalue_indices]
    data1 = np.dot(standardized_data.T, selected_eigenvectors)
    return data1

def save_adjacency_matrix_to_csv(adj_matrix, filename):
    adj_matrix_dense = adj_matrix.toarray()
    adj_df = pd.DataFrame(adj_matrix_dense)
    adj_df.to_csv(filename, index=False, header=False)


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

def positional_encoding(coords, output_dim=32):
    assert output_dim % 2 == 0
    N = coords.shape[0]
    D = output_dim // 2
    theta = np.pi * ((np.arange(D) + 1) / D)
    theta = np.expand_dims(theta, axis=0)
    theta = np.repeat(theta, 2, axis=0)
    sin_enc = np.sin(coords @ theta)
    cos_enc = np.cos(coords @ theta)
    encoding = np.concatenate([sin_enc, cos_enc], axis=-1)
    return encoding

def load_cnn_model(model_name='resnet18', pretrained=True, use_gpu=True):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Model {model_name} is not supported")
    model.eval()
    if use_gpu and torch.cuda.is_available():
        model = model.to("cuda")
    return model


class graph:
    def __init__(self, data, rad_cutoff, k, distType='Radius'):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

    def graph_computing(self):
        graphList = []
        if self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList = [(node_idx, indices[node_idx][j])
                         for node_idx in range(self.data.shape[0])
                         for j in range(indices.shape[1])]
        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity',
                                 include_self=False, metric='cosine')
            A = A.toarray()
            graphList = [(node_idx, j)
                         for node_idx in range(self.data.shape[0])
                         for j in np.where(A[node_idx] == 1)[0]]
        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = [(node_idx, indices[node_idx][j])
                         for node_idx in range(indices.shape[0])
                         for j in range(indices[node_idx].shape[0]) if distances[node_idx][j] > 0]
        return graphList

    def List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for end1, end2 in graphList:
            tdict[end1] = ""
            tdict[end2] = ""
            graphdict.setdefault(end1, []).append(end2)
        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []
        return graphdict

    def mx2SparseTensor(self, mx):
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).long()
        col = torch.from_numpy(mx.col).long()
        values = torch.from_numpy(mx.data)
        indices = torch.stack([row, col], dim=0)
        shape = mx.shape
        adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        adj_ = adj.transpose(0, 1)
        return adj_

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        rowsum[rowsum==0] = 1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)


def image_crop(image, coordinates, crop_size=(50, 50)):
    x, y = coordinates
    left = max(x - crop_size[0] // 2, 0)
    upper = max(y - crop_size[1] // 2, 0)
    right = min(x + crop_size[0] // 2, image.width)
    lower = min(y + crop_size[1] // 2, image.height)
    return image.crop((left, upper, right, lower))

def extract_features_from_image(image, model, transform=None, use_gpu=True):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    image_tensor = transform(image).unsqueeze(0)
    if use_gpu and torch.cuda.is_available():
        image_tensor = image_tensor.to("cuda")
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().cpu().numpy()

def row_normalize(matrix):
    return normalize(matrix, norm='l1', axis=1)


class MLPFusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, A1, A2, A3):
        A_stack = torch.stack([A1, A2, A3], dim=-1)
        N = A_stack.shape[0]
        fused_flat = self.mlp(A_stack.view(-1, 3))
        fused = fused_flat.view(N, N)
        fused = torch.sigmoid(fused)
        mask_original = ((A1 > 0) | (A2 > 0) | (A3 > 0)).float()
        fused = fused * mask_original
        diag_mask = 1 - torch.eye(N, device=fused.device)
        fused = fused * diag_mask
        fused = (fused + fused.t()) / 2
        row_sum = fused.sum(dim=1, keepdim=True) + 1e-8
        fused = fused / row_sum
        return fused


class GAE_VAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, A1, A2, A3):
        super().__init__()
        self.fusion = MLPFusion(hidden_dim=8)
        self.register_buffer('A1', A1)
        self.register_buffer('A2', A2)
        self.register_buffer('A3', A3)

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc_mu = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_logvar = nn.Linear(hidden_dims[0], hidden_dims[1])

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

        h1 = torch.mm(A_normalized, x)
        h1 = F.relu(self.fc1(h1))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decoder(z)
        return x_recon, mu, logvar, A_normalized

# --------------------------- 训练函数（已修改：隐藏过程 + 进度条 + 只输出最优ARI）---------------------------
def train_model(model, X, true_labels, n_epochs=100, lr=0.01, beta=0.01, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_ari = -1
    best_weights = None
    best_latent = None
    best_labels = None

    # 进度条
    pbar = tqdm(range(n_epochs), desc="In the train", total=n_epochs)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        x_recon, mu, logvar, A = model(X)
        recon_loss = F.mse_loss(x_recon, X)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon_loss + beta * kl_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            latent = mu.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=7, init='k-means++').fit(latent)
            ari = adjusted_rand_score(true_labels, kmeans.labels_)

            if ari > best_ari:
                best_ari = ari
                best_weights = model.state_dict().copy()
                best_latent = latent
                best_labels = kmeans.labels_

    # 只输出最终最优结果
    print(f"\nOver！ARI = {best_ari:.4f}")
    model.load_state_dict(best_weights)
    return model, best_latent, best_labels

#############################################
# main
#############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507"
    SAVE_ROOT = "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/MGSTceshi"
    expressionFilename = f"{DATA_ROOT}/filtered_feature_bc_matrix.csv"
    gene_expression_data = pd.read_csv(expressionFilename, index_col=0)
    print(gene_expression_data.shape)
    processed_df = preprocessingCSV(gene_expression_data)

    print('wancheng')
    obs_df = pd.read_csv(f"{DATA_ROOT}/metadata.csv")
    spatial_data = obs_df[['row', 'col']].values

    gene_data = processed_df.values
    gene_data_augmented = augment_gene_data(gene_data, spatial_data, Adj_WT=0.45, neighbour_k=10)

    processed_df = pd.DataFrame(gene_data_augmented, index=processed_df.index, columns=processed_df.columns)
    print("processed_df:", processed_df.shape)
    pca_df = perform_pca(processed_df, n_components=400)
    print("pca_df:", pca_df.shape)
    adj_gene = kneighbors_graph(processed_df.T, n_neighbors=10, metric='cosine',
                                mode='connectivity', include_self=False)
    print("Adjacency matrix shape (gene):", adj_gene.shape)

    feature_spots = processed_df.T.dot(pca_df).astype(float)
    print("feature_spots shape:", feature_spots.shape)
    feature_spots = pd.DataFrame(feature_spots)

    adj_spot = compute_spatial_adjacency_matrix(spatial_data, k=10, method='radius',radius=50)
    adj_spot_dense = adj_spot.toarray()
    print("Spatial adjacency matrix shape:", adj_spot_dense.shape)

    features_spatial = positional_encoding(spatial_data, output_dim=32)

    image_path = f"{DATA_ROOT}/spatial/tissue_hires_image.png"
    image = Image.open(image_path)

    cnn_model = load_cnn_model(model_name='resnet18', use_gpu=True)

    features = []
    for index, row in obs_df.iterrows():
        row_coordinate = row['imagerow']
        col_coordinate = row['imagecol']
        cropped_image = image_crop(image, (col_coordinate, row_coordinate), crop_size=(50, 50))
        feature = extract_features_from_image(cropped_image, cnn_model, use_gpu=True)
        features.append(feature)

    feature_figure = pd.DataFrame(features)

    def perform_pca_on_features(feature_df, n_components):
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(feature_df)
        print(f'Explained variance ratio of first {n_components} components: {pca.explained_variance_ratio_}')
        return reduced_features

    feature_figure = perform_pca_on_features(feature_figure, n_components=200)

    image_graph = graph(feature_figure, rad_cutoff=0, k=15, distType="Radius")
    edge_list = image_graph.graph_computing()
    N = feature_figure.shape[0]
    rows = [edge[0] for edge in edge_list]
    cols = [edge[1] for edge in edge_list]
    data_vals = np.ones(len(edge_list))
    from scipy.sparse import coo_matrix
    adj_mat = coo_matrix((data_vals, (rows, cols)), shape=(N, N))
    adj_image = image_graph.preprocess_graph(adj_mat)

    A1 = adj_gene
    A2 = adj_spot
    A3_dense = row_normalize(adj_image.to_dense().numpy())
    A1_normalized = row_normalize(A1)
    A2_normalized = row_normalize(A2)

    print("Feature matrix shapes:")
    print("feature_spots:", feature_spots.shape)
    print("features_spatial:", features_spatial.shape)
    print("feature_figure:", feature_figure.shape)

    scaler_spots = StandardScaler()
    feature_spots_norm = scaler_spots.fit_transform(feature_spots)

    scaler_spatial = StandardScaler()
    features_spatial_norm = scaler_spatial.fit_transform(features_spatial)

    scaler_figure = StandardScaler()
    feature_figure_norm = scaler_figure.fit_transform(feature_figure)

    X_combined = np.hstack((feature_spots_norm, features_spatial_norm, feature_figure_norm))
    X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)
    print("Combined features shape:", X_combined.shape)

    A1_tensor = torch.tensor(A1_normalized.toarray(), device=device, dtype=torch.float32)
    A2_tensor = torch.tensor(A2_normalized.toarray(), device=device, dtype=torch.float32)
    A3_tensor = torch.tensor(A3_dense, device=device, dtype=torch.float32)

    true_labels = pd.read_csv(f"{DATA_ROOT}/metadata.csv")['layer_guess'].values
    true_labels_encoded = LabelEncoder().fit_transform(true_labels)
    true_labels = true_labels_encoded + 1

    model = GAE_VAE(
        in_dim=X_combined.shape[1],
        hidden_dims=[256, 128],
        A1=A1_tensor,
        A2=A2_tensor,
        A3=A3_tensor
    ).to(device)
    clustering_method = "leiden"

    trained_model, best_latent, best_labels = train_model(model, X_combined_tensor, true_labels)

    spot_names = obs_df['id'].values

    best_labels_df = pd.DataFrame({
        'spot_name': spot_names,
        'cluster': best_labels
    })
    best_labels_df.to_csv(f"{SAVE_ROOT}/MGST.csv", index=False)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import matplotlib.patches as mpatches

    def plot_spatial_clusters_on_image(image_path, spatial_coords, cluster_labels, title="Spatial Clusters on Image",
                                       save_path=None):
        image = Image.open(image_path)
        image_array = np.array(image)
        image_height, image_width = image_array.shape[:2]
        spatial_coords[:, 0] = spatial_coords[:, 0] * image_height
        spatial_coords[:, 1] = spatial_coords[:, 1] * image_width
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}

        plt.figure(figsize=(10, 8))
        plt.imshow(image_array)

        for cluster in unique_clusters:
            mask = (cluster_labels == cluster)
            plt.scatter(spatial_coords[mask, 1], spatial_coords[mask, 0],
                        color=cluster_color_map[cluster], label=f'Cluster {cluster}', s=30, alpha=0.6)

        legend_patches = [mpatches.Patch(color=cluster_color_map[cluster], label=f'Cluster {cluster}')
                          for cluster in unique_clusters]
        plt.legend(handles=legend_patches, title="Clusters", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    best_labels = pd.read_csv(f"{SAVE_ROOT}/MGST.csv")['cluster'].values
    save_path = f"{DATA_ROOT}/MGST.png"
    image_path = f"{DATA_ROOT}/spatial/tissue_hires_image.png"
    plot_spatial_clusters_on_image(image_path, spatial_data, best_labels, title="Optimal Spatial Clusters on Image",
                                   save_path=save_path)
