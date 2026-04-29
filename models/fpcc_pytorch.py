"""PyTorch FPCC backbone/head (simple readable version) for migration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointMLP(nn.Module):
    def __init__(self, in_dim=6, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x):
        # x: [B,N,C]
        return self.net(x)


class FPCCNetTorch(nn.Module):
    def __init__(self, in_dim=6, feat_dim=128):
        super().__init__()
        self.backbone = SimplePointMLP(in_dim=in_dim, feat_dim=feat_dim)
        self.center_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, points):
        # points: [B,N,C]
        feat = self.backbone(points)
        center_logits = self.center_head(feat).squeeze(-1)
        center_score = torch.sigmoid(center_logits)

        # pairwise feature distance as similarity matrix logits
        dist_feat = torch.cdist(feat, feat, p=2) ** 2
        return {
            "point_features": feat,
            "center_logits": center_logits,
            "center_score": center_score,
            "simmat": dist_feat,
        }


def fpcc_loss_torch(net_output, group_one_hot, center_labels, margin_same=0.5, margin_diff=1.0):
    """Approximate FPCC loss in PyTorch.

    group_one_hot: [B,N,G]
    center_labels: [B,N]
    """
    sim = net_output["simmat"]
    B, N, _ = sim.shape

    group_mat = torch.matmul(group_one_hot, group_one_hot.transpose(1, 2))
    eye = torch.eye(N, device=sim.device).unsqueeze(0).expand(B, -1, -1)
    group_mat = torch.maximum(group_mat, eye)

    same = group_mat
    diff = 1.0 - group_mat

    same_loss = 2.0 * same * torch.relu(sim - margin_same)
    diff_loss = diff * torch.relu(margin_diff - sim)
    simmat_loss = (same_loss + diff_loss).mean()

    center_loss = F.smooth_l1_loss(net_output["center_score"], center_labels)
    total = simmat_loss + 3.0 * center_loss
    return total, center_loss, simmat_loss
