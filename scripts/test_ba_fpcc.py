"""Test BA-FPCC-OBB in PyTorch and run FPCC-style clustering + optional OBB."""

import argparse
import glob
import json
import os
import numpy as np
import torch

from models.fpcc_pytorch import FPCCNetTorch
from models.boundary_branch import BoundaryBranch
from utils.test_utils import GroupMerging_fpcc
from utils.obb_utils import estimate_obbs_for_instances
from utils.simple_visualize import visualize_instances_with_obbs


def samples(data, sample_num_point):
    n, dim = data.shape
    # keep original order to avoid prediction-index mismatch
    bnum = int(np.ceil(n / float(sample_num_point)))
    out = np.zeros((bnum, sample_num_point, dim), dtype=np.float32)
    for i in range(bnum):
        beg = i * sample_num_point
        end = min((i + 1) * sample_num_point, n)
        num = end - beg
        out[i, :num] = data[beg:end]
        if num < sample_num_point:
            idx = np.random.choice(n, sample_num_point - num)
            out[i, num:] = data[idx]
    return out


def reassign_boundary_points(points, pred_instance_labels, boundary_scores, threshold=0.5):
    labels = pred_instance_labels.copy()
    bmask = boundary_scores > threshold
    valid = (~bmask) & (labels >= 0)
    if valid.sum() == 0:
        return labels
    centers = {}
    for i in np.unique(labels[valid]):
        centers[int(i)] = points[(labels == i) & valid].mean(axis=0)
    if len(centers) == 0:
        return labels
    cids = np.array(list(centers.keys()))
    cpts = np.array([centers[i] for i in cids])
    idx = np.where(bmask)[0]
    if idx.size == 0:
        return labels
    d2 = ((points[idx, None, :] - cpts[None, :, :]) ** 2).sum(axis=2)
    nn = d2.argmin(axis=1)
    labels[idx] = cids[nn]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoint/ba_fpcc_torch.pt")
    ap.add_argument("--test_glob", type=str, default="datas/ring_test/*.txt")
    ap.add_argument("--point_dim", type=int, default=6)
    ap.add_argument("--point_num", type=int, default=4096)
    ap.add_argument("--center_score_th", type=float, default=0.6)
    ap.add_argument("--r_nms", type=float, default=0.1)
    ap.add_argument("--use_boundary_branch", action="store_true")
    ap.add_argument("--boundary_th", type=float, default=0.5)
    ap.add_argument("--use_obb", action="store_true")
    ap.add_argument("--save_obb", action="store_true")
    ap.add_argument("--visualize_obb", action="store_true")
    ap.add_argument("--obb_json", type=str, default="test_results/ba_fpcc_obb.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPCCNetTorch(in_dim=args.point_dim, feat_dim=128).to(device)
    bbranch = BoundaryBranch(128, 64, input_format="BNC").to(device) if args.use_boundary_branch else None

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    if bbranch is not None and state.get("boundary_branch") is not None:
        bbranch.load_state_dict(state["boundary_branch"])

    model.eval()
    if bbranch is not None:
        bbranch.eval()

    os.makedirs("test_results", exist_ok=True)
    obb_rows = []

    test_files = sorted(glob.glob(args.test_glob))
    if len(test_files) == 0:
        raise FileNotFoundError(f"No test files matched: {args.test_glob}")

    for fp in test_files:
        scene_id = os.path.basename(fp).split(".")[0]
        raw = np.loadtxt(fp)
        pts_xyz = raw[:, :3]
        ins_gt = raw[:, -1].astype(np.int32)

        pts = np.zeros((pts_xyz.shape[0], 6), dtype=np.float32)
        pts[:, :3] = pts_xyz
        pts[:, 3:6] = pts_xyz - pts_xyz.min(axis=0, keepdims=True)

        batches = samples(pts, args.point_num)
        with torch.no_grad():
            x = torch.from_numpy(batches[:, :, : args.point_dim]).float().to(device)
            out = model(x)
            feat = out["point_features"].cpu().numpy().reshape(-1, 128)[: pts.shape[0]]
            cscore = out["center_score"].cpu().numpy().reshape(-1, 1)[: pts.shape[0]]
            bscore = None
            if bbranch is not None:
                blogits = bbranch(out["point_features"])
                bscore = torch.sigmoid(blogits).cpu().numpy().reshape(-1)[: pts.shape[0]]

        pred_ins, _ = GroupMerging_fpcc(pts[:, 3:6], feat, cscore, center_socre_th=args.center_score_th, r_nms=args.r_nms)
        pred_ins = pred_ins.astype(np.int32)
        if bscore is not None:
            pred_ins = reassign_boundary_points(pts[:, :3], pred_ins, bscore, threshold=args.boundary_th)

        if args.use_obb:
            obb_list = estimate_obbs_for_instances(pts[:, :3], pred_ins)
            if args.save_obb:
                for o in obb_list:
                    obb_rows.append({
                        "scene_id": scene_id,
                        "instance_id": int(o["instance_id"]),
                        "center": np.asarray(o["center"]).tolist(),
                        "extent": np.asarray(o["extent"]).tolist(),
                        "R": np.asarray(o["R"]).tolist(),
                        "yaw": float(o["yaw"]),
                        "num_points": int(o["num_points"]),
                    })
            if args.visualize_obb:
                visualize_instances_with_obbs(pts[:, :3], pred_ins, obb_list)

        out_txt = os.path.join("test_results", f"{scene_id}_pred.txt")
        np.savetxt(out_txt, np.column_stack([pts[:, :3], pred_ins]), fmt="%.6f %.6f %.6f %d")
        print("saved pred:", out_txt, "gt instances:", len(np.unique(ins_gt)), "pred instances:", len(np.unique(pred_ins)))

    if args.use_obb and args.save_obb:
        with open(args.obb_json, "w", encoding="utf-8") as f:
            json.dump(obb_rows, f, indent=2)
        print("saved obb:", args.obb_json)


if __name__ == "__main__":
    main()
