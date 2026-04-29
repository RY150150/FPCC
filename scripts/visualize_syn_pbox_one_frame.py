"""Visualize one converted SYN-PBOX frame with instances/boundary/centers/optional OBB."""

import argparse
import numpy as np

from utils.obb_utils import estimate_obbs_for_instances
from utils.simple_visualize import visualize_instances_with_obbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--show_obb", action="store_true")
    args = ap.parse_args()

    data = np.load(args.npz)
    points = data["points"]
    instance_labels = data["instance_labels"]
    boundary_labels = data["boundary_labels"] if "boundary_labels" in data else None
    center_labels = data["center_labels"] if "center_labels" in data else None

    obb_list = estimate_obbs_for_instances(points, instance_labels) if args.show_obb else []
    visualize_instances_with_obbs(points, instance_labels, obb_list, boundary_labels, center_labels)


if __name__ == "__main__":
    main()
