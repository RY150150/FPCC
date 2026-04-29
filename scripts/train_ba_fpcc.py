"""Train BA-FPCC-OBB in PyTorch using original FPCC dataset format."""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.fpcc_dataset import FPCCDataset
from models.fpcc_pytorch import FPCCNetTorch, fpcc_loss_torch
from models.boundary_branch import BoundaryBranch
from losses.boundary_loss import boundary_bce_loss
from utils.boundary_utils import generate_boundary_label


def build_boundary_labels(points_bnc, instance_labels_bn):
    b, n, _ = points_bnc.shape
    out = np.zeros((b, n), dtype=np.float32)
    for i in range(b):
        out[i] = generate_boundary_label(points_bnc[i], instance_labels_bn[i], k=16, ignore_label=-1)
    return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_list", type=str, default="datas/ring_train.txt")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--point_dim", type=int, default=6)
    ap.add_argument("--num_groups", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_path", type=str, default="checkpoint/ba_fpcc_torch.pt")
    ap.add_argument("--use_boundary_loss", action="store_true")
    ap.add_argument("--use_boundary_branch", action="store_true")
    ap.add_argument("--lambda_boundary", type=float, default=1.0)
    ap.add_argument("--lambda_bce", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
    ds = FPCCDataset(args.input_list, point_dim=args.point_dim, num_groups=args.num_groups)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = FPCCNetTorch(in_dim=args.point_dim, feat_dim=128).to(device)
    bbranch = BoundaryBranch(128, 64, input_format="BNC").to(device) if args.use_boundary_branch else None

    params = list(model.parameters()) + (list(bbranch.parameters()) if bbranch is not None else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        if bbranch is not None:
            bbranch.train()
        running = 0.0
        for batch in dl:
            points = batch["points"].float().to(device)
            group_one_hot = batch["group_one_hot"].float().to(device)
            center_labels = batch["center_labels"].float().to(device)
            instance_labels = batch["instance_labels"].cpu().numpy().astype(np.int32)

            out = model(points)
            total_loss, center_loss, sim_loss = fpcc_loss_torch(out, group_one_hot, center_labels)

            boundary_labels = None
            if args.use_boundary_loss or args.use_boundary_branch:
                boundary_labels = build_boundary_labels(points.detach().cpu().numpy()[:, :, :3], instance_labels).to(device)

            if args.use_boundary_loss:
                weights = 1.0 + args.lambda_boundary * boundary_labels
                weighted_center = ((out["center_score"] - center_labels).abs() * weights).mean()
                total_loss = total_loss + weighted_center

            if args.use_boundary_branch:
                logits = bbranch(out["point_features"])
                bce = boundary_bce_loss(logits, boundary_labels)
                total_loss = total_loss + args.lambda_bce * bce

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            running += float(total_loss.item())

        print(f"Epoch {epoch+1}/{args.epochs} loss={running/max(len(dl),1):.6f}")

    torch.save({"model": model.state_dict(), "boundary_branch": None if bbranch is None else bbranch.state_dict()}, args.save_path)
    print("Saved checkpoint:", args.save_path)


if __name__ == "__main__":
    main()
