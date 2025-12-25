import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import scipy.sparse as sp




def load_cnn_model(model_name='resnet18', pretrained=True, use_gpu=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Model {model_name} is not supported")
    model.eval()
    if use_gpu and torch.cuda.is_available():
        model = model.to("cuda")
    return model


#############################################

#############################################
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
            # Use cosine distance here
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
        # Use PyTorch's built-in sparse_coo_tensor instead of torch-sparse
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).long()
        col = torch.from_numpy(mx.col).long()
        values = torch.from_numpy(mx.data)
        indices = torch.stack([row, col], dim=0)
        shape = mx.shape
        adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        # Transpose the sparse tensor
        adj_ = adj.transpose(0, 1)
        return adj_

    def preprocess_graph(self, adj):
        # Normalize sparse matrix using SciPy (without self-loops)
        adj = sp.coo_matrix(adj)
        # Do not add self-loops: comment out the line below
        adj = adj + sp.eye(adj.shape[0])
        # Use original adjacency matrix directly
        rowsum = np.array(adj.sum(1))
        # Prevent division by zero
        rowsum[rowsum==0] = 1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)


def image_crop(image, coordinates, crop_size=(50, 50)):
    x, y = coordinates  # Assume coordinates are (row, col)
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

