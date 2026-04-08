# UI Scene Semantic Project

基于 [PRD_跨应用移动界面语义相似度建模与场景聚类系统.md](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\PRD_跨应用移动界面语义相似度建模与场景聚类系统.md) 的轻量多模态移动界面语义项目。

项目当前围绕 [data_qwen_split](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\data_qwen_split) 构建完整实验闭环：

- 图像单模态分类
- 图像 + UI JSON 多模态分类
- embedding 导出
- 聚类与原型页分析

## 技术栈

- `Python 3.10+`
- `PyTorch`
- `torchvision`
- `NumPy`
- `PyYAML`
- `scikit-learn`
- `Pillow`
- `matplotlib`
- `pytest`
- `PowerShell`

## 项目结构

- `configs/`: 实验配置
- `docs/`: 架构与设计文档
- `src/ui_scene/`: 核心代码
- `scripts/`: 训练与分析入口
- `tests/`: 单元测试
- `outputs/`: checkpoint、embedding、聚类结果

## 数据目录

- `dataset_split/`: 原始截图划分
- `data_qwen/`: 原始截图 + JSON 按类组织
- `data_qwen_split/`: 当前训练使用的数据目录

## 如何跑起来

### 1. 准备环境

推荐使用独立虚拟环境，并确保版本为 `Python 3.10+`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你的 PowerShell 默认禁止脚本执行，可以临时打开一个 bypass 会话：

```powershell
powershell -ExecutionPolicy Bypass
```

### 2. 准备数据

当前训练默认使用 [data_qwen_split](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\data_qwen_split)，其结构应为：

```text
data_qwen_split/
├─ train/
├─ val/
├─ test/
└─ manifests/
```

如果你还只有 [data_qwen](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\data_qwen) 和 [dataset_split](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\dataset_split)，可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_data_qwen_split.ps1 -CleanOutput -LinkMode HardLink
```

### 3. 运行测试

```powershell
pytest
```

### 4. 查看配置

主要配置文件：

- [base.yaml](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\configs\base.yaml)
- [train_image.yaml](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\configs\train_image.yaml)
- [train_multimodal.yaml](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\configs\train_multimodal.yaml)
- [retrieval.yaml](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\configs\retrieval.yaml)

### 5. 训练图像单模态模型

完整训练：

```powershell
python .\scripts\train_image.py
```

快速 smoke test：

```powershell
python .\scripts\train_image.py --epochs 1 --batch-size 8 --limit-train 64 --limit-val 32
```

### 6. 训练多模态模型

完整训练：

```powershell
python .\scripts\train_multimodal.py
```

快速 smoke test：

```powershell
python .\scripts\train_multimodal.py --epochs 1 --batch-size 4 --limit-train 16 --limit-val 8
```

### 7. 导出 embedding

以某个 checkpoint 为输入，导出指定 split 的 embedding：

```powershell
python .\scripts\extract_embeddings.py --checkpoint .\outputs\runs\train_image_baseline_xxx\best_model.pt --split test
```

多模态模型同理：

```powershell
python .\scripts\extract_embeddings.py --checkpoint .\outputs\runs\train_multimodal_baseline_xxx\best_model.pt --split test
```

### 8. 运行聚类分析

```powershell
python .\scripts\run_clustering.py --embeddings .\outputs\embeddings\train_multimodal_baseline_xxx_best_model\test.npz --num-clusters 22
```

## 当前开发状态

已完成：

- PRD 文档
- 数据集重组脚本
- `data_qwen_split` 训练目录
- 系统架构设计
- 图像单模态训练脚本
- 多模态训练脚本
- embedding 导出脚本
- 聚类与原型页分析脚本

下一步建议：

- 增加检索脚本，直接展示 Top-K 相似页面
- 优化 UI 树线性化规则
- 为多模态模型加入更稳定的树编码方式
- 增加可视化分析，如 t-SNE / UMAP

## 相关文档

- [PRD_跨应用移动界面语义相似度建模与场景聚类系统.md](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\PRD_跨应用移动界面语义相似度建模与场景聚类系统.md)
- [SYSTEM_ARCHITECTURE.md](d:\HuaweiMoveData\Users\曾景麟\Desktop\apppredict\docs\SYSTEM_ARCHITECTURE.md)
