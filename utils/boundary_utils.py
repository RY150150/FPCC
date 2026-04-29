"""Boundary label generation utilities for BA-FPCC-OBB."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_boundary_label(points, instance_labels, k=16, ignore_label=-1):
    """Generate binary boundary labels from instance ids.

    Args:
        points (np.ndarray): [N, 3]
        instance_labels (np.ndarray): [N]
        k (int): number of nearest neighbors
        ignore_label (int): ignored instance id, boundary set to 0

    Returns:
        np.ndarray: boundary labels [N], values in {0, 1}
    """
    points = np.asarray(points)
    instance_labels = np.asarray(instance_labels)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be [N,3], got {points.shape}")
    if instance_labels.ndim != 1 or instance_labels.shape[0] != points.shape[0]:
        raise ValueError("instance_labels shape must be [N] and match points")

    n = points.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.uint8)

    k_eff = min(max(2, k), n)
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(points)
    indices = nn.kneighbors(points, return_distance=False)

    boundary = np.zeros((n,), dtype=np.uint8)
    for i in range(n):
        cur_label = instance_labels[i]
        if cur_label == ignore_label:
            continue
        nbr_labels = instance_labels[indices[i]]
        nbr_labels = nbr_labels[nbr_labels != ignore_label]
        if nbr_labels.size == 0:
            continue
        if np.any(nbr_labels != cur_label):
            boundary[i] = 1
    return boundary


if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.randn(200, 3) * 0.02 + np.array([0.0, 0.0, 0.0])
    b = np.random.randn(200, 3) * 0.02 + np.array([0.05, 0.0, 0.0])
    pts = np.vstack([a, b])
    ins = np.array([0] * 200 + [1] * 200)
    bd = generate_boundary_label(pts, ins, k=16)
    print("Total points:", len(pts))
    print("Boundary points:", int(bd.sum()))
