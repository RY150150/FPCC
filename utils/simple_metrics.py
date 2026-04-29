"""Simple metrics for BA-FPCC-OBB experiments."""

import time
import numpy as np
import torch


def compute_iou(pred_mask, gt_mask):
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_precision_recall_f1(pred_labels, gt_labels):
    pred = np.asarray(pred_labels).reshape(-1)
    gt = np.asarray(gt_labels).reshape(-1)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def compute_boundary_f1(pred_boundary, gt_boundary):
    _, _, f1 = compute_precision_recall_f1(pred_boundary, gt_boundary)
    return f1


def compute_center_error(pred_center, gt_center):
    pred = np.asarray(pred_center)
    gt = np.asarray(gt_center)
    return float(np.linalg.norm(pred - gt))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_fps(model, dataloader, num_batches=20):
    model.eval()
    use_cuda = next(model.parameters()).is_cuda if any(True for _ in model.parameters()) else False
    t0 = time.time()
    cnt = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if use_cuda:
                x = x.cuda(non_blocking=True)
            _ = model(x)
            cnt += x.shape[0]
    dt = max(time.time() - t0, 1e-8)
    return float(cnt / dt)


if __name__ == "__main__":
    print("iou:", compute_iou([1, 1, 0], [1, 0, 0]))
    print("f1:", compute_precision_recall_f1([1, 0, 1], [1, 1, 0]))
    # TODO: mAP for instance segmentation if needed in future.
