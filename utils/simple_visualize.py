"""Simple Open3D visualization helpers for instances, boundaries, centers and OBBs."""

import numpy as np
import open3d as o3d


def _color_by_instance(instance_labels):
    uniq = np.unique(instance_labels)
    color_map = {}
    rng = np.random.RandomState(0)
    for u in uniq:
        if u < 0:
            color_map[u] = np.array([0.5, 0.5, 0.5])
        else:
            color_map[u] = rng.rand(3)
    colors = np.array([color_map[i] for i in instance_labels], dtype=np.float64)
    return colors


def visualize_instances_with_obbs(points, instance_labels, obb_list, boundary_labels=None, center_labels=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(_color_by_instance(instance_labels))

    geoms = [pcd]

    if boundary_labels is not None:
        bmask = boundary_labels.astype(bool)
        if bmask.any():
            bpcd = o3d.geometry.PointCloud()
            bpcd.points = o3d.utility.Vector3dVector(points[bmask])
            bpcd.paint_uniform_color([1.0, 0.0, 0.0])
            geoms.append(bpcd)

    if center_labels is not None:
        cmask = center_labels > 0.5
        if cmask.any():
            cpcd = o3d.geometry.PointCloud()
            cpcd.points = o3d.utility.Vector3dVector(points[cmask])
            cpcd.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append(cpcd)

    for obb_item in obb_list:
        obb = o3d.geometry.OrientedBoundingBox(
            center=np.array(obb_item["center"]),
            R=np.array(obb_item["R"]),
            extent=np.array(obb_item["extent"]),
        )
        obb.color = (0.0, 0.0, 0.0)
        geoms.append(obb)

    o3d.visualization.draw_geometries(geoms)
