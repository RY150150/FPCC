"""Convert one SYN-PBOX frame to BA-FPCC npz with boundary labels."""

import argparse
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np

from utils.boundary_utils import generate_boundary_label


def find_masks(mask_dir, frame_id):
    patterns = [
        os.path.join(mask_dir, f"{frame_id:06d}_*.png"),
        os.path.join(mask_dir, f"{frame_id:04d}_*.png"),
        os.path.join(mask_dir, f"{frame_id}_*.png"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(list(set(files)))
    if not files:
        raise FileNotFoundError(f"No mask found for frame_id={frame_id} in {mask_dir}")
    return files


def backproject_depth(depth, K, depth_scale):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))  # u: x(pixel col), v: y(pixel row)
    z = depth.astype(np.float32) / float(depth_scale)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_camera", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--frame_id", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_points", type=int, default=4096)
    args = ap.parse_args()

    with open(args.scene_camera, "r", encoding="utf-8") as f:
        scene_camera = json.load(f)

    frame_key = str(args.frame_id)
    if frame_key not in scene_camera:
        raise KeyError(f"frame_id={args.frame_id} not found in scene_camera")

    cam = scene_camera[frame_key]
    K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)
    depth_scale = float(cam.get("depth_scale", 1000.0))

    depth = imageio.imread(args.depth)
    pts_img = backproject_depth(depth, K, depth_scale)

    masks = find_masks(args.mask_dir, args.frame_id)
    instance_map = np.full(depth.shape, -1, dtype=np.int32)
    for ins_id, mpath in enumerate(masks):
        m = imageio.imread(mpath)
        instance_map[m > 0] = ins_id

    valid = depth > 0
    points = pts_img[valid]
    instance_labels = instance_map[valid]

    if points.shape[0] == 0:
        raise RuntimeError("No valid depth points after filtering depth>0")

    n = points.shape[0]
    if n >= args.num_points:
        idx = np.random.choice(n, args.num_points, replace=False)
    else:
        idx = np.random.choice(n, args.num_points, replace=True)

    points = points[idx]
    instance_labels = instance_labels[idx]

    center_labels = np.zeros((args.num_points,), dtype=np.float32)
    for ins in np.unique(instance_labels):
        if ins < 0:
            continue
        ins_idx = np.where(instance_labels == ins)[0]
        c = points[ins_idx].mean(axis=0)
        near = ins_idx[np.argmin(np.linalg.norm(points[ins_idx] - c, axis=1))]
        center_labels[near] = 1.0

    boundary_labels = generate_boundary_label(points, instance_labels, k=16, ignore_label=-1).astype(np.float32)

    np.savez(args.output, points=points, instance_labels=instance_labels, center_labels=center_labels, boundary_labels=boundary_labels)
    print("Saved:", args.output)
    for ins in np.unique(instance_labels):
        print(f"instance {ins}: {(instance_labels==ins).sum()} points")


if __name__ == "__main__":
    main()
