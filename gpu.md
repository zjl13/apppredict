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

### E8. Suggestion-driven objective reshaping (`CE + SupCon + domain adversarial`)
目的：按 `建议.md` 的高优先级方向，在不更换当前 best backbone / fusion 路线的前提下，测试“跨 app 语义对齐”是否能直接带来分类增益。

本轮改动
- 在 fused embedding 上加入 `Supervised Contrastive Loss`
- 新增轻量 `domain adversarial head`，通过 `GRL` 约束表示尽量弱化 app / package domain 信息
- domain 定义采用从 tree text 中提取的 coarse package name，保留 `top-128` 高频 domain，其余归入 `__other__`
- 训练从当前 best `train_multimodal_resnet34_refine_20260409_103229` checkpoint 继续微调
- 保持 `resnet34 + semantic_v3 + hybrid_hashing(2048) + gated fusion + EMA + 288 分辨率` 不变，避免把结果混到 backbone 变化里

结果
- Run: `outputs/runs/train_multimodal_resnet34_supcon_domain_20260409_164635`
- Best val acc / macro-F1: `0.6614 / 0.5832`
- 这个 best 实际出现在 `epoch 0` 的 resume eval，之后 `epoch 1-6` 持续回落，最终到 `0.6531 / 0.5679`
- 相比当前 best `train_multimodal_resnet34_refine_20260409_103229`：`-0.0002 acc`，`-0.0003 macro-F1`

结论
- `建议.md` 里“目标函数改造”的总体方向值得关注，但当前这版轻量实现没有带来分类提升。
- 在现有采样与 domain 定义下，`SupCon + GRL` 更像是额外约束，拉低了主任务收敛效率。
- 当前可以确认的是：不要直接复用这套配置和权重指望拿到更高 accuracy。
- 如果后续还要重试这条路线，应同时引入更贴任务的正负样本构造或 app-aware batch sampler，而不是只把两个 loss 直接挂上去。

### E9. ResNet50 stronger-backbone attempt from ImageNet init
目的：验证 `gpu.md` 中“更强视觉 backbone”这条路线，看看仅通过提升图像主干容量，能否在当前多模态配方上继续推高 accuracy。

本轮改动
- 在代码里补充 `resnet50` backbone 支持，保持现有 `semantic_v3 + hybrid_hashing(2048) + gated fusion` 不变
- 使用新的 `configs/train_multimodal_resnet50_acc.yaml`
- 训练从 ImageNet 预训练初始化开始，不使用旧 `resnet34` checkpoint 续训
- 使用 `batch_size=16 / image_size=288 / EMA / backbone_lr_scale=0.25 / freeze_backbone_epochs=1`

结果
- Run: `outputs/runs/train_multimodal_resnet50_acc_20260409_171605`
- Best val acc / macro-F1: `0.5938 / 0.4889`
- 12 个 epoch 正常跑完，没有进程异常或 OOM
- 最佳点出现在 `epoch 12`，说明它还在学习，但当前配置下明显落后于现有 best `0.6615 / 0.5834`

结论
- 这次不是“训练挂了”，而是“这套 resnet50 从头接多模态头的配置效果不够好”。
- 因为完整 12 epoch 已经跑完且明显落后，所以不值得重启同一配置。
- 如果后续还要重试更强 backbone，应该优先调整训练日程或迁移方式，而不是原样重跑这版 `resnet50` 配置。

### E10. ResNet34 refine with too little regularization
目的：验证当前 best checkpoint 在后续精调阶段，是否已经不再需要 auxiliary heads loss 和 branch dropout 这类额外正则。

本轮改动
- 从当前 best `train_multimodal_resnet34_refine_20260409_103229` checkpoint 继续训练
- 将 `auxiliary_loss` 从 `0.05 / 0.05` 直接降到 `0.0 / 0.0`
- 将 `branch_dropout` 从 `0.01` 降到 `0.0`
- 同时把学习率压到 `3e-5`，尝试做一次更“干净”的二阶段 refine

结果
- Run: `outputs/runs/train_multimodal_resnet34_refine_noreg_20260409_183451`
- Resume eval 的 best val acc / macro-F1: `0.6614 / 0.5832`
- 实际训练到 `epoch 1` 就掉到 `0.6550 / 0.5726`，`epoch 2` 继续掉到 `0.6525 / 0.5711`
- 因为连续两轮都明显差于 resume baseline，所以提前停止，没有继续浪费 GPU

结论
- 当前 best 路线还不能把辅助监督和分支级正则一起拿掉。
- 这类“去正则精调”会让续训很快过拟合，至少在当前数据和学习率区间下不是有效方向。
- 如果还要继续 refine，更合理的是保留原有结构，只做更温和的学习率和时长调整。

### E11. ResNet34 gentle stage-2 refine
目的：验证是否能在完全保留当前 best 结构与正则的前提下，仅靠更低学习率和更短续训，再从 best checkpoint 挤出一点额外收益。

本轮改动
- 从当前 best `train_multimodal_resnet34_refine_20260409_103229` checkpoint 继续训练
- 保留 `auxiliary_loss=0.05 / 0.05`、`branch_dropout=0.01`、`dropout=0.10`
- 只把学习率降到 `2e-5`，把 `backbone_lr_scale` 降到 `0.15`，总 epoch 缩到 `4`

结果
- Run: `outputs/runs/train_multimodal_resnet34_refine_stage2_20260409_185223`
- Resume eval 的 best val acc / macro-F1: `0.6614 / 0.5832`
- `epoch 1` 降到 `0.6559 / 0.5749`，`epoch 2` 继续到 `0.6540 / 0.5749`
- 因为前两轮都没有接近或超过当前 best，所以提前停止

结论
- 当前 best checkpoint 周围，单纯靠更小学习率的二阶段续训也没有带来提升。
- 这说明 `train_multimodal_resnet34_refine_20260409_103229` 附近已经比较接近局部最优，继续做“续训挤分”性价比很低。
- 下一步应优先回到有新信息增量的方向，例如更强的 tree 表征，而不是围绕同一 checkpoint 反复微调。

### E12. Revisit 4096-dim hybrid tree with persistent cache
目的：重新验证当初因为预处理成本太高而搁置的 `4096` 维 hybrid tree 路线，并确认在持久化缓存已经成熟后，它到底是“工程上跑不动”还是“建模上没价值”。

本轮改动
- 保持 `resnet34 + semantic_v3 + gated fusion` 不变，主要变量只放在 tree 表征上
- 将 tree vectorizer 从 `hybrid_hashing(2048)` 提升到 `hybrid_hashing(4096)`，即 `word_dim=2048 + char_dim=2048`
- 依赖现在已经跑通的 tree text / tree feature 持久化缓存与并行预处理能力
- 训练配置回到更接近 `resnet34_acc` 的从头训练路线，而不是继续围绕当前 best checkpoint 做小幅续训

结果
- Run: `outputs/runs/train_multimodal_resnet34_hybrid4096_revisit_20260409_190439`
- Best val acc / macro-F1: `0.6732 / 0.5991`
- 最佳点出现在 `epoch 12`
- 相比上一版 best `train_multimodal_resnet34_refine_20260409_103229`：`+0.0117 acc`，`+0.0157 macro-F1`
- Embedding: `outputs/embeddings/train_multimodal_resnet34_hybrid4096_revisit_20260409_190439_best_model`
- Clustering: `NMI 0.4767 / ARI 0.3737 / Silhouette 0.2120`

结论
- `4096` 维 hybrid tree 并不是方向错了，之前的主要问题确实是工程链路还不够成熟。
- 在持久化缓存可用之后，这条路线不仅能完整跑通，而且带来了当前最好的分类结果。
- 这轮也说明：相比围绕旧 checkpoint 反复做低风险续训，给 tree branch 增加新的可用信息增量更有效。

## What To Avoid Repeating
- 不要再参考 2026-04-06 的 `1.0 / 1.0` 结果。
- 不要把“只调训练器”误认为“多模态专项优化”；它有帮助，但不是决定性因素。
- 不要重复尝试过于简单的 `shallow tree encoder + plain concat fusion` 方案，它已经验证过上限有限。
- 在没有持久化缓存前，不要默认上 `4096` 维 hybrid tree 特征；启动成本太高，迭代效率不划算。
- 如果目标优先是 overall accuracy，不要默认启用 `weighted_random sampler`；它更偏向照顾长尾类，不一定提高总体准确率。
- 对高分辨率 refine 路线，不要默认把续训 epoch 拉太长；当前 best 出现在 `epoch 2`，后续主要是缓慢过拟合。
- 不要直接重复 `train_multimodal_resnet34_supcon_domain` 这套 `SupCon(weight=0.08) + domain adversarial(weight=0.03, top-128 coarse domains)` 配置；它没有超过 resume baseline，且训练后期持续掉点。
- 不要原样重跑 `train_multimodal_resnet50_acc` 这套 `12 epoch + ImageNet init + freeze_backbone_epochs=1` 配置；它完整跑完后仍明显低于当前 best。
- 不要把当前 best refine 路线的 `auxiliary_loss` 和 `branch_dropout` 一次性全部拿掉；这会让续训在前两轮就明显掉点。
- 不要原样重跑基于当前 best checkpoint 的“更低学习率 stage-2 refine”方案；它也没有带来提升。

## Current Best Result
- Best image model: `train_image_optimized_20260407_194729`
  - `val acc 0.5723 / macro-F1 0.5009`
- Best multimodal classification model: `train_multimodal_resnet34_hybrid4096_revisit_20260409_190439`
  - `val acc 0.6732 / macro-F1 0.5991`
- Best multimodal clustering result by NMI / ARI: `train_multimodal_resnet34_hybrid4096_revisit_20260409_190439_best_model`
  - `NMI 0.4767 / ARI 0.3737 / Silhouette 0.2120`
- Best multimodal clustering silhouette so far: `train_multimodal_resnet34_acc_20260408_190505_best_model`
  - `Silhouette 0.2221`

## Next Likely Directions
优先级从高到低：
1. 在新的 `4096` best checkpoint 上补一轮真正轻量的低学习率 refine，验证它是否还能在更强 tree 表征基础上继续往上走。
2. 增加独立 `test` 分类评估脚本，导出每类 precision / recall / F1 和 confusion matrix 摘要，方便答辩与定向调参。
3. 继续优化当前 `memmap + DataLoader workers` 的数据路径，降低 `4096` 路线的 CPU-bound 程度，提高后续迭代效率。
4. 如果后续再碰更强视觉 backbone，优先考虑更合理的迁移或更长 schedule，而不是原样重跑这版从 ImageNet 直接起步的 `resnet50` 配置。
5. 如果再回到“跨 app 语义对齐”方向，优先补 app-aware batch sampler 或更严格的跨 domain 正样本构造，而不是原样重跑当前这版 `SupCon + GRL`。
