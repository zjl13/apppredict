# GPU Experiment Log

## Goal
在不做大规模平台化改造的前提下，稳定完成并迭代下面这条闭环：
1. 图像单模态分类训练
2. 图像 + UI JSON 多模态分类训练
3. 导出 test embedding
4. 对 embedding 做聚类分析
5. 沉淀实验记录，方便复现、答辩和避免重复尝试

## Workflow Rule
- 从现在开始，只要出现“当前最好”的结果，就同步做三件事：
  1. 更新本文件中的实验记录
  2. 说明本轮具体做了什么优化
  3. 提交一次 git commit
- commit 只提交代码、配置、文档和实验结论，不提交 `data_qwen_split/` 与 `outputs/` 这类本地大文件目录。

## Environment Snapshot
- Date: 2026-04-08 UTC
- Python: `3.12.3`
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 5090`
- Data check: `data_qwen_split/{train,val,test,manifests}` 存在
- Test status: `python -m pytest` 通过，`1 passed`

## Invalid Or Discarded Attempts
### 2026-04-06 smoke/minimal runs
- 现象：历史上出现过 `1.0 / 1.0` 的异常高分。
- 结论：这是最小化数据或错误流程产物，不可信，不能作为正式结果。
- 处理：对应旧 run 和旧 embedding 已删除，不再参考。

### 2026-04-07 early image run `train_image_baseline_20260407_165542`
- 结果：`val acc 0.3749`，`macro-F1 0.3073`
- 结论：这是中间启动的早期图像训练，效果明显弱于后续完整 baseline，不作为正式基线。

## Completed Experiments
### E1. Full baseline closure
目的：先完成真实全量数据上的最小闭环，建立可信 baseline。

图像单模态 baseline
- Run: `outputs/runs/train_image_baseline_20260407_173553`
- Best val acc / macro-F1: `0.4981 / 0.4333`
- Embedding: `outputs/embeddings/train_image_baseline_20260407_173553_best_model`
- Clustering: `NMI 0.3057 / ARI 0.1397 / Silhouette 0.0583`

多模态 baseline
- Run: `outputs/runs/train_multimodal_baseline_20260407_184831`
- Best val acc / macro-F1: `0.5138 / 0.4505`
- Embedding: `outputs/embeddings/train_multimodal_baseline_20260407_184831_best_model`
- Clustering: `NMI 0.3358 / ARI 0.1512 / Silhouette 0.0728`

结论
- 全量 baseline 是可信的，明显低于错误的 `1.0 / 1.0`。
- 多模态相对图像 baseline 有小幅优势，但优势还不够大。

### E2. Generic training optimization
目的：先用低风险训练层优化，看看能否稳定提升图像和多模态。

本轮改动
- 引入 `AdamW`
- 引入 `weighted_random` sampler
- 引入 cosine scheduler
- 增加 `label_smoothing`
- 启用 AMP
- 支持 `--resume-from` 从 baseline best checkpoint 继续微调
- 树线性化增加缓存
- DataLoader 改成更适合 GPU 的 `num_workers` / `pin_memory` 设置

图像优化版
- Run: `outputs/runs/train_image_optimized_20260407_194729`
- Best val acc / macro-F1: `0.5723 / 0.5009`
- 相比图像 baseline：`+0.0742 acc`，`+0.0676 macro-F1`
- Clustering: `NMI 0.3540 / ARI 0.1931 / Silhouette 0.0573`

多模态优化版
- Run: `outputs/runs/train_multimodal_optimized_20260407_200829`
- Best val acc / macro-F1: `0.5666 / 0.5051`
- 相比多模态 baseline：`+0.0528 acc`，`+0.0545 macro-F1`
- Clustering: `NMI 0.3810 / ARI 0.2331 / Silhouette 0.0941`

结论
- 这组优化有效，说明训练器、采样与 AMP 的组合值得保留。
- 但它主要属于“通用训练优化”，图像和多模态都会受益。
- 仅靠这组改动，不能显著放大多模态相对图像的独特优势。

### E3. Multimodal-specific architecture optimization
目的：不再只调训练器，直接增强树分支表达能力和融合方式。

本轮改动
- 树特征从默认浅语义，升级为 `semantic_v2` profile
- 树输入维度提高到 `2048`，减少 hashing 冲突
- 树编码器从单层浅 MLP 升级为带 `LayerNorm + GELU` 的 deep encoder
- 融合方式从简单 concat 改为 `gated fusion`
- 在融合中加入 `abs(image - tree)` 差异特征
- 训练配置单独沉淀到 `configs/train_multimodal_gated.yaml`

多模态 gated 最佳版
- Run: `outputs/runs/train_multimodal_gated_20260407_210705`
- Best val acc / macro-F1: `0.6342 / 0.5723`
- 相比多模态优化版：`+0.0676 acc`，`+0.0672 macro-F1`
- 相比多模态 baseline：`+0.1203 acc`，`+0.1218 macro-F1`
- Embedding: `outputs/embeddings/train_multimodal_gated_20260407_210705_best_model`
- Clustering: `NMI 0.4428 / ARI 0.3373 / Silhouette 0.1872`

结论
- 这轮首次显著拉开了多模态和图像基线的差距。
- 真正拉开差距的不是继续调 optimizer，而是增强树语义特征和 fusion 结构。
- 当时的最佳路线：`semantic_v2 + tree_input_dim=2048 + deep tree encoder + gated fusion`。

### E4. High-dim hybrid tree attempt (`4096` dims)
目的：进一步提升树分支表达能力，把多模态信息做得更细。

本轮改动
- 引入 `semantic_v3`，增加 `package / focusable / long-clickable / adapter-view / draw / depth / child_count / coarse spatial buckets`
- 引入 `hybrid_hashing`，把词级和字符级 hashing 特征拼起来
- 引入 `resnet18` 图像主干、辅助分类头、branch dropout
- 给数据侧增加 tree text cache、tree feature precompute 和并行预处理能力

结果
- `tree_input_dim=4096` 的全量 run 在启动阶段预处理成本过高，未形成值得保留的完整有效结果。
- 结论：在没有持久化磁盘缓存前，`4096` 维 hybrid tree 特征当前性价比太低，不适合作为默认路线。

### E5. ResNet18 fast hybrid (`2048` dims)
目的：保留 `semantic_v3 + hybrid hashing` 的核心收益，同时把预处理和训练速度拉回可迭代区间。

本轮改动
- 把 tree feature 缩到 `2048` 维
- 加入 tree text / tree feature 并行预处理
- 保留 `resnet18 + gated fusion + aux heads`

结果
- Run: `outputs/runs/train_multimodal_resnet_fast_20260408_182736`
- Best val acc / macro-F1: `0.6224 / 0.5700`
- 结论：这轮没有超过当时 best `0.6342`，但证明了并行预处理链路是有效的，后续更强 backbone 可以直接复用。

### E6. ResNet34 accuracy-focused multimodal
目的：围绕“整体 accuracy”而不是长尾宏平均，做更贴近目标函数的优化。

本轮改动
- 图像 backbone 升级为 `resnet34`
- 取消 `weighted_random sampler`，改回自然分布训练
- 取消 `label_smoothing`
- 降低 `dropout / branch_dropout / auxiliary loss` 强度
- 保留 `semantic_v3 + hybrid_hashing(2048) + gated fusion + 并行预处理`

结果
- Run: `outputs/runs/train_multimodal_resnet34_acc_20260408_190505`
- Best val acc / macro-F1: `0.6591 / 0.5780`
- 相比上一版最佳 gated 多模态：`+0.0249 acc`，`+0.0057 macro-F1`
- Embedding: `outputs/embeddings/train_multimodal_resnet34_acc_20260408_190505_best_model`
- Clustering: `NMI 0.4637 / ARI 0.3671 / Silhouette 0.2221`

结论
- 当前最佳路线已经从 `mobilenet_v3_small` 切换到 `resnet34`。
- 为了冲 accuracy，取消重采样和过强正则是有效的。
- 当前最好结果仍然离 `80% accuracy` 有明显差距，后续核心矛盾变成：更强视觉 backbone、持久化树缓存、以及是否需要更强的文本/结构编码器。

### E7. ResNet34 refine with EMA + higher resolution
目的：在当前 best checkpoint 基础上做低学习率精调，看看更高分辨率和更稳的优化策略能否继续往上挤 accuracy。

本轮改动
- 从 `train_multimodal_resnet34_acc_20260408_190505` 的 best checkpoint 继续训练
- 图像尺寸从 `256` 提到 `288`
- 引入 `EMA`
- 给 image backbone 使用 `0.35x` 学习率，head 保持基础学习率
- 引入 tree text / tree feature 持久化磁盘缓存
- 进一步下调 `auxiliary loss / dropout / branch_dropout`

结果
- Run: `outputs/runs/train_multimodal_resnet34_refine_20260409_103229`
- Best val acc / macro-F1: `0.6615 / 0.5834`
- 相比上一版最佳 `train_multimodal_resnet34_acc_20260408_190505`：`+0.0024 acc`，`+0.0054 macro-F1`
- Best 出现在 `epoch 2`，后续 epoch 缓慢回落

结论
- 这轮是有效提升，但幅度不大，更像是“精调挤分”而不是路线级跃迁。
- 持久化树缓存已经跑通，后续同类配置不需要每轮都重算 tree text / tree feature。
- 当前 `memmap + 多 worker + 288 分辨率` 组合仍然偏 CPU-bound，且最佳点出现很早；后续应优先做更轻的数据路径或 early stopping，而不是盲目拉长 epoch。

## What To Avoid Repeating
- 不要再参考 2026-04-06 的 `1.0 / 1.0` 结果。
- 不要把“只调训练器”误认为“多模态专项优化”；它有帮助，但不是决定性因素。
- 不要重复尝试过于简单的 `shallow tree encoder + plain concat fusion` 方案，它已经验证过上限有限。
- 在没有持久化缓存前，不要默认上 `4096` 维 hybrid tree 特征；启动成本太高，迭代效率不划算。
- 如果目标优先是 overall accuracy，不要默认启用 `weighted_random sampler`；它更偏向照顾长尾类，不一定提高总体准确率。
- 对高分辨率 refine 路线，不要默认把续训 epoch 拉太长；当前 best 出现在 `epoch 2`，后续主要是缓慢过拟合。

## Current Best Result
- Best image model: `train_image_optimized_20260407_194729`
  - `val acc 0.5723 / macro-F1 0.5009`
- Best multimodal classification model: `train_multimodal_resnet34_refine_20260409_103229`
  - `val acc 0.6615 / macro-F1 0.5834`
- Best multimodal clustering result: `train_multimodal_resnet34_acc_20260408_190505_best_model`
  - `NMI 0.4637 / ARI 0.3671 / Silhouette 0.2221`

## Next Likely Directions
优先级从高到低：
1. 先把当前 best refine checkpoint 补跑 embedding / clustering，确认分类提升是否也能带来表示空间收益。
2. 优先优化当前 `memmap + DataLoader workers` 的数据路径，降低 CPU-bound 程度；缓存已经有了，下一步要把读取链路也做轻。
3. 在当前 accuracy 路线下继续尝试更强视觉 backbone，例如 `resnet50` 或更强的现代 backbone；当前瓶颈仍然更像视觉表达能力。
4. 增加独立 `test` 分类评估脚本，导出每类 precision / recall / F1 和 confusion matrix 摘要，方便答辩与定向调参。
