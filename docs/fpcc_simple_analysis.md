# FPCC 简要结构分析（第一步，仅阅读）

> 说明：原始项目是 **TensorFlow 1.x** 实现，不是 PyTorch。若要在 BA-FPCC-OBB 中统一到 PyTorch，需要新增脚本做兼容/迁移。

1. **训练入口文件**
   - `fpcc_train.py`。

2. **测试入口文件**
   - `fpcc_test.py`。

3. **模型 forward 输入/输出**
   - 输入来自 `models/model.py::get_model(backbone, point_cloud, is_training, ...)`。
   - `point_cloud` shape 约为 `[B, N, POINT_DIM]`（训练中一般取前 6 维）。
   - 输出是 dict：
     - `center_score`: 每点中心置信度，shape 约 `[B, N]`
     - `point_features`: 每点特征（Fsim），shape 约 `[B, N, 128]`
     - `simmat`: 相似度矩阵，shape `[B, N, N]`（train=True 时）
     - `3d_distance`: 点间 3D 距离矩阵，shape `[B, N, N]`

4. **dataloader 返回字段**
   - `provider.loadDataFile_with_groupseglabel_stanfordindoor(...)` 在训练里被调用，返回：
     - `cur_data`
     - `cur_group`
     - `_`（未使用）
     - `_`（未使用）
     - `cur_score`

5. **每个 batch 点云 shape**
   - 训练 feed 到模型的是 `input_data[..., :POINT_DIM]`，shape `[B, 4096, POINT_DIM]`。

6. **instance label 与 center label shape**
   - instance label（group label）在 one-hot 后为 `ptsgroup_label_ph`: `[B, N, NUM_GROUPS]`。
   - center label 为 `pts_score_ph`: `[B, N]`。

7. **loss 在哪里计算**
   - 在 `models/model.py::get_loss(...)` 计算。
   - 在 `fpcc_train.py` 中通过 `loss, score_loss, grouperr = model.get_loss(...)` 调用。

8. **聚类在哪里做**
   - 在 `fpcc_test.py` 中调用 `utils.test_utils` 里的 `GroupMerging_fpcc(...)` 完成实例聚类。

9. **加 boundary loss 建议改哪里**
   - 最小侵入：新增训练脚本 `scripts/train_ba_fpcc.py`，在原始 loss 基础上额外做逐点加权。
   - 逐点 loss 可以优先作用在 center 回归项（`ptscenter_loss`）或新增点级监督项。

10. **加 boundary branch 建议改哪里**
   - 最小侵入：不要改原 `fpcc_train.py`，在新脚本里拿 `net_output['point_features']` 后接一个轻量 BoundaryBranch。
   - 若 TF 图里插分支过于复杂，可先在 PyTorch 训练流程里演示并保留 TODO。

11. **加 OBB 后处理从哪里拿预测实例点云**
   - 在 `fpcc_test.py` 聚类后得到 `ins_pre`（每点实例 id）。
   - 用 `pts[:,0:3]`（或对齐后的 `pts[:,3:6]`）按实例 id 取点，再做 OBB。
