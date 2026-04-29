"""Open3D based OBB estimation utilities for BA-FPCC-OBB."""

import numpy as np
import open3d as o3d


def estimate_obb(points):
    """Estimate oriented bounding box for points [N,3]."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {points.shape}")
    if points.shape[0] < 4:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    center = np.asarray(obb.center)
    extent = np.asarray(obb.extent)
    R = np.asarray(obb.R)
    # TODO: yaw 依赖坐标系定义，这里按 Z-up 假设从 R 提取平面角。
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return {"center": center, "extent": extent, "R": R, "yaw": yaw}


def estimate_obbs_for_instances(points, instance_labels):
    points = np.asarray(points)
    instance_labels = np.asarray(instance_labels)
    out = []
    for ins_id in np.unique(instance_labels):
        if ins_id < 0:
            continue
        mask = instance_labels == ins_id
        ins_points = points[mask]
        obb = estimate_obb(ins_points)
        if obb is None:
            continue
        obb["instance_id"] = int(ins_id)
        obb["num_points"] = int(ins_points.shape[0])
        out.append(obb)
    return out


if __name__ == "__main__":
    np.random.seed(0)
    pts = np.random.rand(1000, 3) * np.array([0.4, 0.2, 0.1])
    obb = estimate_obb(pts)
    print("obb:", obb)
