import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from config import DEVICE


def get_centroids_and_labels(layer, n_clusters):
    """
    Extracts kernel centroids and corresponding labels
    for 1st conv weights of provided layer (block).

    Parameters
    ----------
    layer : BasicBlock of ResNet20
        block to get 1st conv weights centroids from
    n_clusters : int
        number of clusters

    Returns
    -------
    torch.Tensor
        clusters' centers
    numpy.ndarray
        labels of each point
    """
    km, _ = get_kmeans(layer, n_clusters)
    size = list(layer.conv1.weight.shape[-3:])
    centroids = km.cluster_centers_.reshape(n_clusters, *size)
    centroids = torch.from_numpy(centroids).to(DEVICE)
    return centroids, km.labels_


def get_mask_from_centroid_labels_(labels):
    """
    Creates mask for unique labels

    Parameters
    ----------
    labels : numpy.ndarray
        labels of corresponding centroids (from get_centroids_and_labels)

    Returns
    -------
    numpy.ndarray
        mask of shape (len(labels), )
    """
    n = []
    for label in labels:
        if label not in n:
            n.append(label)
        else:
            n.append(None)
    size = len(labels)
    mask = np.zeros(size)
    for i in range(size):
        mask[i] = 0 if n[i] is None else 1

    return mask


def get_kmeans(layer, n_clusters):
    """
    Utilizes sklearn.cluster._kmeans.KMeans

    Parameters
    ----------
    layer : BasicBlock of ResNet20
        block to get 1st conv weights centroids from
    n_clusters : int
        number of clusters (number of centroids to generate)

    Returns
    -------
    sklearn.cluster._kmeans.KMeans
        KMeans class instance fitted on provided data
    numpy.ndarray
        1st conv weights flatten from dim 1
    """
    weights = layer.conv1.weight
    weights_flatten = weights.flatten(1).detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(weights_flatten)
    return km, weights_flatten


def get_mask_from_closest(layer, n_clusters, mask4d=False):
    """
    Creates mask for conv1 filter weights of provided layer which are closest
    to centroid for each cluster (according to the euclidean distance).

    Parameters
    ----------
    layer : BasicBlock of ResNet20
        block to get 1st conv weights centroids from
    n_clusters : int
        number of clusters (number of centroids to generate)
    mask4d : bool, default=False
        whether to create a mask of size (layer.conv1.weight.shape)

    Returns
    -------
    numpy.ndarray (if mask4d=False)
        mask of size (layer.conv1.weight.shape[0])
    torch.Tensor (if mask4d=True)
        mask of size (layer.conv1.weight.shape)
    """
    km, weights_flatten = get_kmeans(layer, n_clusters)
    ids, dists = \
        pairwise_distances_argmin_min(km.cluster_centers_, weights_flatten)

    size = layer.conv1.weight.shape
    mask = np.arange(size[0])

    for i in mask:
        mask[i] = 1 if i in ids else 0

    if mask4d:
        mask = get_mask4d(mask, size)

    return mask


def get_mask4d(mask1d, size):
    """
    Creates a mask of provided size from 1-dimensional mask

    Parameters
    ----------
    mask1d : numpy.ndarray
        1-dimensional mask to transform
    size : torch.Size or tuple or list
        size of transformed mask

    Returns
    -------
    torch.Tensor
        mask of size (size)
    """
    assert mask1d.shape[0] == size[0]
    mask = torch.zeros(size)

    for i, n in enumerate(mask1d):
        mask[i].zero_() if n == 0 else mask[i].zero_().add_(1)

    return mask.to(DEVICE)


def drop_weights_(layer, mask):
    """
    Prunes (inplace) weights in accordance with provided mask

    Parameters
    ----------
    layer : BasicBlock of ResNet20
        block to prune 1st conv weights from
    mask : numpy.ndarray
        1-dimensional mask to prune weights in accordance with

    Returns
    -------
    None
    """
    n_channels_left = int(sum(mask))

    layer.conv1.weight = nn.Parameter(layer.conv1.weight[mask == 1, :, :, :])
    layer.conv1.out_channels = n_channels_left

    layer.bn1.num_features = n_channels_left
    layer.bn1.running_mean = layer.bn1.running_mean[mask == 1]
    layer.bn1.running_var = layer.bn1.running_var[mask == 1]
    layer.bn1.weight = nn.Parameter(layer.bn1.weight[mask == 1])
    layer.bn1.bias = nn.Parameter(layer.bn1.bias[mask == 1])

    layer.conv2.in_channels = n_channels_left
    layer.conv2.weight = nn.Parameter(layer.conv2.weight[:, mask == 1, :, :])

    return None


def prune_replace(layer, n_clusters, replace_weights=True):
    """
    Prunes (inplace) weights which are not in the most closest to centroid for each cluster,
    and, if replace_weights=True, replaces the closest ones with corresponding centroids.

    Parameters
    ----------
    layer : BasicBlock of ResNet20
        block to prune 1st conv weights from
    n_clusters : int
        number of clusters (number of centroids to generate)
    replace_weights : bool, default=True
        whether to replace weights with centroids of corresponding clusters

    Returns
    -------
    None
    """
    centroids, labels = get_centroids_and_labels(layer, n_clusters)
    mask = get_mask_from_centroid_labels_(labels)

    if replace_weights:
        for i, label in enumerate(labels):
            layer.conv1.weight.data[i] = centroids[label]

    drop_weights_(layer, mask)

    return None
