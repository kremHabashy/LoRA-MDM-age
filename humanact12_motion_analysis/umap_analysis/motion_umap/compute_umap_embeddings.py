import torch
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from umap import UMAP

def extract_motion_features(joint_positions, method='flatten', reduce_dims=50):
    '''
    joint_positions: Tensor of shape (N, T, J, 3)
    method: 'flatten' or 'mean'
    Returns: (N, feature_dim)
    '''
    if method == 'flatten':
        features = joint_positions.reshape(joint_positions.size(0), -1)
    elif method == 'mean':
        features = joint_positions.mean(dim=1).reshape(joint_positions.size(0), -1)
    else:
        raise ValueError("Unknown feature extraction method")
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features.cpu().numpy())
    return features

def compute_umap(features, labels, n_neighbors=15, min_dist=0.1, n_components=3):
    '''
    features: np.array (N, D)
    labels: list or array of length N
    Returns: 3D UMAP embeddings
    '''
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    embeddings = reducer.fit_transform(features)
    return embeddings

def plot_umap(embeddings, labels, title="Interactive 3D UMAP of Motion Features", save_path=None):
    '''
    embeddings: (N, 3)
    labels: list or array of class labels
    '''
    fig = px.scatter_3d(
        x=embeddings[:, 0], 
        y=embeddings[:, 1], 
        z=embeddings[:, 2],
        color=[str(label) for label in labels],
        title=title,
        labels={"color": "Class"}
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    if save_path:
        fig.write_html(save_path)
    fig.show()