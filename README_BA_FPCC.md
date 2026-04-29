# BA-FPCC-OBB

本 README 是对当前仓库里 BA-FPCC-OBB 改造的**完整运行说明**，目标是：
- 不破坏原始 FPCC（TensorFlow）流程；
- 提供可独立运行的 PyTorch 迁移流程；
- 支持 Boundary Label / Boundary Loss / Boundary Branch / Boundary-aware 后处理 / OBB / SYN-PBOX 单帧转换与可视化。

---

## 0. 分支与安全策略（强烈推荐）

为了确保原始 FPCC 不受影响，请在独立分支开发与验证：

```bash
git checkout -b feature/ba-fpcc-pytorch-migration
```

- 原始 TF 入口：`fpcc_train.py`、`fpcc_test.py`
- 新增 PyTorch 入口：`scripts/train_ba_fpcc.py`、`scripts/test_ba_fpcc.py`

---

## 1. 目录说明（新增模块）

- `datasets/fpcc_dataset.py`：读取原 FPCC h5 列表文件，输出 PyTorch batch 字段。
- `models/fpcc_pytorch.py`：PyTorch FPCC 主干 + FPCC 风格损失。
- `models/boundary_branch.py`：边界预测分支。
- `losses/boundary_loss.py`：boundary 加权损失与 BCE 损失。
- `utils/boundary_utils.py`：基于 kNN 的 boundary label 自动生成。
- `utils/obb_utils.py`：Open3D OBB 估计。
- `utils/simple_visualize.py`：实例/边界/中心/OBB 可视化。
- `utils/simple_metrics.py`：简化指标函数。
- `scripts/train_ba_fpcc.py`：PyTorch 训练入口（含 BA 开关）。
- `scripts/test_ba_fpcc.py`：PyTorch 测试入口（含后处理与 OBB 开关）。
- `scripts/convert_syn_pbox_one_frame.py`：SYN-PBOX 单帧转换。
- `scripts/visualize_syn_pbox_one_frame.py`：SYN-PBOX 单帧可视化。

---

## 2. 环境依赖

> 你当前机器如果提示 `ModuleNotFoundError`，先执行这一节。

### 2.1 PyTorch 分支依赖

```bash
pip install numpy scipy h5py scikit-learn imageio open3d
pip install torch torchvision torchaudio
```

### 2.2 原始 FPCC（TensorFlow）依赖

原仓库是 TensorFlow 1.x 风格代码，建议单独环境安装：

```bash
pip install numpy scipy h5py
pip install tensorflow==1.15.5
```

> 若你的系统无法安装 TF1，请只跑 PyTorch 分支，原始分支作为结构参考。

---

## 3. 数据格式说明

### 3.1 PyTorch 训练数据（与原 FPCC 一致）

`--input_list` 指向一个 txt 文件（如 `datas/ring_train.txt`），每行是一个 h5 文件路径。h5 至少包含：
- `data`：点云输入
- `pid` 或 `gid`：实例 id
- `score` 或 `center_score`：中心监督标签

### 3.2 PyTorch 测试数据

默认读取 `datas/ring_test/*.txt`，每行格式近似：

```text
x y z instance_id
```

---

## 4. 运行命令

## 4.1 原始 FPCC baseline（TF）

```bash
python fpcc_train.py --gpu 0 --input_list datas/ring_train.txt
python fpcc_test.py --gpu 0 --restore_dir checkpoint/
```

## 4.2 PyTorch BA-FPCC 训练

```bash
python scripts/train_ba_fpcc.py \
  --input_list datas/ring_train.txt \
  --batch_size 4 \
  --epochs 100 \
  --point_dim 6 \
  --num_groups 50 \
  --use_boundary_loss \
  --use_boundary_branch \
  --lambda_boundary 1.0 \
  --lambda_bce 0.5 \
  --save_path checkpoint/ba_fpcc_torch.pt
```

## 4.3 PyTorch BA-FPCC 测试 + Boundary-aware + OBB

```bash
python scripts/test_ba_fpcc.py \
  --ckpt checkpoint/ba_fpcc_torch.pt \
  --test_glob 'datas/ring_test/*.txt' \
  --point_dim 6 \
  --point_num 4096 \
  --center_score_th 0.6 \
  --r_nms 0.1 \
  --use_boundary_branch \
  --boundary_th 0.5 \
  --use_obb \
  --save_obb \
  --obb_json test_results/ba_fpcc_obb.json
```

## 4.4 SYN-PBOX 单帧转换

```bash
python scripts/convert_syn_pbox_one_frame.py \
  --scene_camera path/to/scene_camera.json \
  --depth path/to/depth.png \
  --mask_dir path/to/mask_visib \
  --frame_id 0 \
  --output demo_syn_frame.npz \
  --num_points 4096
```

## 4.5 SYN-PBOX 单帧可视化

```bash
python scripts/visualize_syn_pbox_one_frame.py --npz demo_syn_frame.npz --show_obb
```

---

## 5. 开关与消融建议

### 5.1 训练开关（`scripts/train_ba_fpcc.py`）
- `--use_boundary_loss`：开启 boundary 加权项。
- `--use_boundary_branch`：开启 boundary branch + BCE。
- `--lambda_boundary`：boundary 加权强度。
- `--lambda_bce`：boundary BCE 损失权重。

建议消融顺序：
1. 仅主干（两个开关都关）
2. 主干 + boundary loss
3. 主干 + boundary branch
4. 全开（boundary loss + boundary branch）

### 5.2 测试开关（`scripts/test_ba_fpcc.py`）
- `--use_boundary_branch`：加载并使用边界分数。
- `--boundary_th`：边界点阈值。
- `--use_obb`：计算 OBB。
- `--save_obb`：保存 OBB json。
- `--visualize_obb`：Open3D 交互可视化。

---

## 6. 输出说明

### 6.1 测试输出
- `test_results/*_pred.txt`：每场景预测实例结果。
- `test_results/ba_fpcc_obb.json`：OBB 列表（当 `--use_obb --save_obb` 开启时）。

OBB 字段包含：
- `scene_id`
- `instance_id`
- `center`
- `extent`
- `R`
- `yaw`
- `num_points`

### 6.2 SYN-PBOX NPZ 输出
`convert_syn_pbox_one_frame.py` 输出字段：
- `points`
- `instance_labels`
- `center_labels`
- `boundary_labels`

---

## 7. 常见错误与排查

### 7.1 `ModuleNotFoundError: numpy / torch / tensorflow`
先安装第 2 节依赖。

### 7.2 `No test files matched`
检查 `--test_glob` 路径与引号是否正确。

### 7.3 OBB 可视化打不开
检查 `open3d` 是否安装；无桌面环境时不要开 `--visualize_obb`，只保存 json。

### 7.4 TF1 安装困难
建议优先使用 PyTorch 分支；原 TF 分支可作为对照基线。

---

## 8. 当前实现边界（实话说明）

- PyTorch 主干是“可读优先”的迁移版本，不是逐层 1:1 复刻 TF 主干。
- 若你要严格复现实验数值，需要继续做：
  1) backbone 细节对齐；
  2) loss 细节对齐；
  3) 数据预处理完全对齐。

但从工程可运行性上，当前分支已经具备完整链路：
**数据读取 -> 训练 -> 测试聚类 -> boundary 后处理 -> OBB 输出 -> 可视化**。
