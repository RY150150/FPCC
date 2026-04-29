# BA-FPCC-OBB

本版本提供 **PyTorch 主干迁移版**（不改坏原 FPCC TensorFlow 代码）。

## 安全建议（强烈推荐）
先在新分支开发，避免影响原始 FPCC：
```bash
git checkout -b feature/ba-fpcc-pytorch-migration
```

## 1) 原始 FPCC baseline（保留不变）
```bash
python fpcc_train.py --gpu 0 --input_list datas/ring_train.txt
python fpcc_test.py --gpu 0 --restore_dir checkpoint/
```

## 2) PyTorch 版 BA-FPCC 训练
```bash
python scripts/train_ba_fpcc.py \
  --input_list datas/ring_train.txt \
  --batch_size 4 \
  --epochs 100 \
  --use_boundary_loss \
  --use_boundary_branch \
  --lambda_boundary 1.0 \
  --lambda_bce 0.5 \
  --save_path checkpoint/ba_fpcc_torch.pt
```

## 3) PyTorch 版 BA-FPCC 测试 + OBB
```bash
python scripts/test_ba_fpcc.py \
  --ckpt checkpoint/ba_fpcc_torch.pt \
  --test_glob 'datas/ring_test/*.txt' \
  --use_boundary_branch \
  --use_obb --save_obb \
  --obb_json test_results/ba_fpcc_obb.json
```

## 4) SYN-PBOX 单帧转换
```bash
python scripts/convert_syn_pbox_one_frame.py \
  --scene_camera path/to/scene_camera.json \
  --depth path/to/depth.png \
  --mask_dir path/to/mask_visib \
  --frame_id 0 \
  --output demo_syn_frame.npz
```

## 5) SYN-PBOX 单帧可视化
```bash
python scripts/visualize_syn_pbox_one_frame.py --npz demo_syn_frame.npz --show_obb
```

## 说明
- 原始 TF 训练/测试入口未删除，便于回归对照。
- 新增 PyTorch 训练/测试脚本已使用原 FPCC 数据格式（h5/txt）并包含 BA 开关。


## 环境依赖
```bash
pip install numpy torch scikit-learn open3d imageio h5py scipy
```
