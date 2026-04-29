"""Simple boundary prediction branch for point features."""

import torch
import torch.nn as nn


class BoundaryBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, input_format="BNC"):
        super().__init__()
        if input_format not in ["BNC", "BCN"]:
            raise ValueError("input_format must be 'BNC' or 'BCN'")
        self.input_format = input_format
        if input_format == "BNC":
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels, 1, 1),
            )

    def forward(self, x):
        if self.input_format == "BNC":
            out = self.net(x).squeeze(-1)
        else:
            out = self.net(x).squeeze(1)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 4096, 128)
    m = BoundaryBranch(128, 64, input_format="BNC")
    y = m(x)
    print("BNC out:", y.shape)

    x2 = x.permute(0, 2, 1)
    m2 = BoundaryBranch(128, 64, input_format="BCN")
    y2 = m2(x2)
    print("BCN out:", y2.shape)
