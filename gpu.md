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
- 这是目前最有效的一轮优化，也是当前最佳结果。
- 真正拉开差距的不是继续调 optimizer，而是增强树语义特征和 fusion 结构。
- 当前最佳路线：`semantic_v2 + tree_input_dim=2048 + deep tree encoder + gated fusion`。

## What To Avoid Repeating
- 不要再参考 2026-04-06 的 `1.0 / 1.0` 结果。
- 不要把“只调训练器”误认为“多模态专项优化”；它有帮助，但不是决定性因素。
- 不要重复尝试过于简单的 `shallow tree encoder + plain concat fusion` 方案，它已经验证过上限有限。

## Current Best Result
- Best image model: `train_image_optimized_20260407_194729`
  - `val acc 0.5723 / macro-F1 0.5009`
- Best multimodal model: `train_multimodal_gated_20260407_210705`
  - `val acc 0.6342 / macro-F1 0.5723`
- Best clustering result: `train_multimodal_gated_20260407_210705_best_model`
  - `NMI 0.4428 / ARI 0.3373 / Silhouette 0.1872`

## Next Likely Directions
优先级从高到低：
1. 增加独立 `test` 分类评估脚本，导出每类 precision / recall / F1 和 confusion matrix 摘要，方便答辩与定向调参。
2. 在当前 gated 多模态结构上继续做小步搜索，例如 `tree_input_dim 4096`、`tree_hidden_dim 384`、`fusion_dim 384`。
3. 针对长尾类和 `Other` 类试 `focal loss` 或更细粒度重采样，但需要和当前 `weighted_random` 分开做消融。
