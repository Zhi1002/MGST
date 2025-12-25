import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from PIL import Image
from sklearn.decomposition import PCA
import torch
from sklearn.neighbors import NearestNeighbors
from spatial import compute_spatial_adjacency_matrix,positional_encoding
from image import load_cnn_model,image_crop,extract_features_from_image,graph
from VAEmodel import GAE_VAE,train_model

def augment_gene_data(data, spatial_coords, Adj_WT, neighbour_k=10):
    # Compute neighbor indices
    knn = NearestNeighbors(n_neighbors=neighbour_k, metric='euclidean')
    knn.fit(spatial_coords)
    neigh_indices = knn.kneighbors(spatial_coords, return_distance=False)  # (4226, 4)

    print("gene_data.shape:", data.shape)  # (2000, 4226)
    print("neigh_indices.shape:", neigh_indices.shape)  # (4226, 4)

    # Compute neighbor averages
    adjacent_avg = np.mean(np.take(data, neigh_indices, axis=1), axis=2)  # (2000, 4226)

    # Ensure consistent dimensions
    print("adjacent_avg.shape:", adjacent_avg.shape)  # (2000, 4226)

    augment_data = data + Adj_WT * adjacent_avg  # Ensure (2000, 4226) + (2000, 4226)

    return augment_data


def preprocessingCSV(df, expressionFilename=None):
    if isinstance(df, pd.DataFrame):
        data = df
    else:
        data = pd.read_csv(expressionFilename, index_col=0, header=0)

    # Remove genes with expression proportion less than 1%
    data = data[data[data.columns[1:]].astype('bool').mean(axis=1) >= 0.01]
    print('After preprocessing, {} genes remaining'.format(data.shape[0] - 1))

    # Select top 2000 genes with highest variance
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
    adj_matrix_dense = adj_matrix.toarray()  # Convert to dense matrix
    adj_df = pd.DataFrame(adj_matrix_dense)  # Convert to DataFrame
    adj_df.to_csv(filename, index=False, header=False)  # Save as CSV file

def row_normalize(matrix):
    return normalize(matrix, norm='l1', axis=1)


#############################################
# Main function entry
#############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess expression matrix
    expressionFilename = "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/expression_matrix.csv"
    gene_expression_data = pd.read_csv(expressionFilename, index_col=0)
    print(gene_expression_data.shape)
    processed_df = preprocessingCSV(gene_expression_data)

    # processed_df.to_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/Hp.csv")
    print('completed')
    # Load spatial coordinate data (ensure rows correspond to gene data)
    obs_df = pd.read_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/metadata.csv")
    spatial_coords = obs_df[['row', 'col']].values

    # Data augmentation: augment gene expression data based on spatial information
    gene_data = processed_df.values
    gene_data_augmented = augment_gene_data(gene_data, spatial_coords, Adj_WT=0.6, neighbour_k=10)
    # Optionally replace original data with augmented data
    processed_df = pd.DataFrame(gene_data_augmented, index=processed_df.index, columns=processed_df.columns)
    print("processed_df:", processed_df.shape)
    pca_df = perform_pca(processed_df, n_components=400)
    print("pca_df:", pca_df.shape)
    # Construct gene adjacency matrix (based on transposed expression data)
    adj_gene = kneighbors_graph(processed_df.T, n_neighbors=10, metric='cosine',
                                mode='connectivity', include_self=False)
    print("Adjacency matrix shape (gene):", adj_gene.shape)

    # Construct spot feature representation
    feature_spots = processed_df.T.dot(pca_df).astype(float)
    print("feature_spots shape:", feature_spots.shape)
    feature_spots = pd.DataFrame(feature_spots)
    feature_spots.to_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/feature_spots.csv",
                           index=False, header=False)

    # Load spatial coordinates
    obs_df = pd.read_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/metadata.csv")
    spatial_data = obs_df[['row', 'col']].values
    print("Spatial data shape:", spatial_data.shape)
    # Build adjacency matrix using improved method
    adj_spot = compute_spatial_adjacency_matrix(spatial_data, k=10, method='radius',radius=50)
    adj_spot_dense = adj_spot.toarray()
    print("Spatial adjacency matrix shape:", adj_spot_dense.shape)

    # Perform high-dimensional positional encoding on spatial coordinates
    features_spatial = positional_encoding(spatial_data, output_dim=32)
    pd.DataFrame(features_spatial).to_csv('E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/features_spatial.csv',
                                            index=False, header=False)

    # Load image file
    image_path = "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/spatial/tissue_hires_image.png"
    image = Image.open(image_path)

    # Load CNN model
    cnn_model = load_cnn_model(model_name='resnet18', use_gpu=True)

    # Extract image features
    features = []
    for index, row in obs_df.iterrows():
        row_coordinate = row['imagerow']
        col_coordinate = row['imagecol']
        cropped_image = image_crop(image, (col_coordinate, row_coordinate), crop_size=(50, 50))
        feature = extract_features_from_image(cropped_image, cnn_model, use_gpu=True)
        features.append(feature)

    feature_figure = pd.DataFrame(features)
    feature_figure.to_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/feature_figure.csv",
                           index=False, header=False)

    def perform_pca_on_features(feature_df, n_components):
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(feature_df)
        print(f'Explained variance ratio of first {n_components} components: {pca.explained_variance_ratio_}')
        return reduced_features

    # Perform PCA dimensionality reduction on extracted image features
    feature_figure = perform_pca_on_features(feature_figure, n_components=300)

    # --------------------- Modified part: Build adjacency matrix for image features using graph class --------------------- #
    image_graph = graph(feature_figure, rad_cutoff=0, k=15, distType="Radius")
    edge_list = image_graph.graph_computing()
    N = feature_figure.shape[0]
    rows = [edge[0] for edge in edge_list]
    cols = [edge[1] for edge in edge_list]
    data_vals = np.ones(len(edge_list))
    from scipy.sparse import coo_matrix
    adj_mat = coo_matrix((data_vals, (rows, cols)), shape=(N, N))
    # Normalize adjacency matrix and convert to sparse tensor (using PyTorch built-in)
    adj_image = image_graph.preprocess_graph(adj_mat)
    # ------------------------------------------------------------------------------------- #

    # Normalize adjacency matrices
    A1 = adj_gene
    A2 = adj_spot
    # For A3: adj_image is sparse tensor, first convert to dense, then normalize
    A3_dense = row_normalize(adj_image.to_dense().numpy())
    A1_normalized = row_normalize(A1)
    A2_normalized = row_normalize(A2)

    save_adjacency_matrix_to_csv(A1_normalized, "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/adj_gene.csv")
    save_adjacency_matrix_to_csv(A2_normalized, "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/adj_spot.csv")
    pd.DataFrame(A3_dense).to_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/adj_image1.csv",
                                  index=False, header=False)

    print("Feature matrix shapes:")
    print("feature_spots:", feature_spots.shape)
    print("features_spatial:", features_spatial.shape)
    print("feature_figure:", feature_figure.shape)

    # Standardize each feature matrix separately
    scaler_spots = StandardScaler()
    feature_spots_norm = scaler_spots.fit_transform(feature_spots)

    scaler_spatial = StandardScaler()
    features_spatial_norm = scaler_spatial.fit_transform(features_spatial)

    scaler_figure = StandardScaler()
    feature_figure_norm = scaler_figure.fit_transform(feature_figure)

    # Horizontally concatenate feature matrices
    X_combined = np.hstack((feature_spots_norm, features_spatial_norm, feature_figure_norm))
    X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)
    print("Combined features shape:", X_combined.shape)

    # Convert normalized adjacency matrices to tensors
    A1_tensor = torch.tensor(A1_normalized.toarray(), device=device, dtype=torch.float32)
    A2_tensor = torch.tensor(A2_normalized.toarray(), device=device, dtype=torch.float32)
    A3_tensor = torch.tensor(A3_dense, device=device, dtype=torch.float32)

    # Load true labels
    true_labels = pd.read_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/metadata.csv")['layer_guess'].values
    true_labels_encoded = LabelEncoder().fit_transform(true_labels)
    true_labels = true_labels_encoded + 1

    # Initialize GAE model with three adjacency matrices
    model = GAE_VAE(
        in_dim=X_combined.shape[1],
        hidden_dims=[256, 128],
        A1=A1_tensor,
        A2=A2_tensor,
        A3=A3_tensor
    ).to(device)
    clustering_method = "leiden"
    # Train model
    trained_model, best_latent, best_labels = train_model(model, X_combined_tensor, true_labels)
    # Get spot names
    spot_names = obs_df['id'].values  # Assuming obs_df index contains spot names
    # If spot names are in a column, change to: spot_names = obs_df['spot_name'].values

    # Save optimal cluster labels as CSV file with spot names in first column
    best_labels_df = pd.DataFrame({
        'spot_name': spot_names,
        'cluster': best_labels
    })
    best_labels_df.to_csv("E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/BayesSpace.csv", index=False)

    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import matplotlib.patches as mpatches


    def plot_spatial_clusters_on_image(image_path, spatial_coords, cluster_labels, title="Spatial Clusters on Image",
                                       save_path=None):
        # Load original image
        image = Image.open(image_path)
        image_array = np.array(image)

        # Get image dimensions
        image_height, image_width = image_array.shape[:2]

        # Scale coordinates
        spatial_coords[:, 0] = spatial_coords[:, 0] * image_height
        spatial_coords[:, 1] = spatial_coords[:, 1] * image_width

        # Get unique clusters and generate colors
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))  # Color mapping
        cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}

        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(image_array)  # Display original image

        # Plot clustering results
        for cluster in unique_clusters:
            mask = (cluster_labels == cluster)
            plt.scatter(spatial_coords[mask, 1], spatial_coords[mask, 0],
                        color=cluster_color_map[cluster], label=f'Cluster {cluster}', s=30, alpha=0.6)

        # Add legend
        legend_patches = [mpatches.Patch(color=cluster_color_map[cluster], label=f'Cluster {cluster}')
                          for cluster in unique_clusters]
        plt.legend(handles=legend_patches, title="Clusters", loc='center left', bbox_to_anchor=(1, 0.5))

        # Figure settings
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')

        # Save or display figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')  # Ensure legend is not cut off
        plt.show()


    # Get spatial coordinates
    spatial_coords = obs_df[['imagerow', 'imagecol']].values  # Get spatial coordinates

    # Plot clustering results on original slice image
    image_path = "E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/spatial/tissue_hires_image.png"
    plot_spatial_clusters_on_image(image_path, spatial_coords, best_labels,
                                   title="Optimal Spatial Clusters on Image",
                                   save_path="E:/jsj/user06/ST/stDCL_data/stDCL/DLPFC/151507/BayesSpace/optimal_spatial_clusters_on_image.png")

