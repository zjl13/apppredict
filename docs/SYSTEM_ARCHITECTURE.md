# 系统架构设计

## 1. 架构目标

本项目采用适合个人大创推进的轻量离线架构，不做重服务化平台，而是围绕“训练、评估、检索、聚类”建立稳定闭环。

核心原则：

- 先把实验闭环跑通，再逐步增强
- 模块边界清晰，便于单人维护
- 数据流简单，避免过早工程化

## 2. 总体架构

系统分为五层：

1. 数据层
2. 预处理层
3. 模型层
4. 训练评估层
5. 检索聚类分析层

## 3. 模块划分

### 3.1 数据层

输入数据来自 `data_qwen_split`：

- `images/*.jpg`
- `jsons/*.json`
- `manifests/*.jsonl`

数据层职责：

- 读取样本路径和标签
- 管理 train/val/test 划分
- 提供统一样本结构

### 3.2 预处理层

预处理层负责把原始截图和 UI 树转成模型可接受的输入。

图像侧：

- resize
- normalize
- augmentation

UI 树侧：

- 提取关键字段
- 遍历树结构
- 线性化节点序列
- 构造节点 token 和属性特征

### 3.3 模型层

模型层采用轻量双分支架构：

- Image Encoder：提取截图视觉特征
- Tree Encoder：提取 UI 树结构语义特征
- Fusion Head：融合两路特征
- Classification Head：输出 22 类场景
- Embedding Head：输出页面语义向量

### 3.4 训练评估层

训练评估层负责：

- 训练循环
- 验证与 early stopping
- 指标统计
- checkpoint 保存

第一阶段以分类任务为主，后续可以逐步加入度量学习损失。

### 3.5 检索聚类分析层

该层基于 embedding 工作：

- 构建页面向量库
- 执行 Top-K 相似检索
- 执行聚类分析
- 选取原型页面

## 4. 核心数据流

1. `manifest` 读取样本记录
2. `Dataset` 加载 `jpg + json + label`
3. 预处理层输出图像张量与树序列特征
4. 多模态模型输出分类 logits 和 embedding
5. 训练层计算 loss 并优化
6. 推理后将 embedding 保存到 `outputs/embeddings`
7. 检索与聚类模块读取 embedding 生成分析结果

## 5. 为什么这样设计

### 5.1 适合个人项目

- 开发路径清晰
- 每个模块都可单独验证
- 不依赖复杂线上服务

### 5.2 适合当前数据

- 当前已有稳定的分类标签
- 已有截图和 JSON 一一对应
- 非常适合从监督分类平滑升级到检索和聚类

### 5.3 适合后续扩展

未来如果要增加：

- 更强的树编码器
- OCR 信息
- 端侧推理导出
- 简易可视化演示

都可以在当前架构上继续迭代，而不需要推翻重来。

## 6. 推荐仓库结构

```text
apppredict/
├─ configs/
├─ docs/
├─ outputs/
├─ scripts/
├─ src/ui_scene/
│  ├─ data/
│  ├─ preprocess/
│  ├─ models/
│  ├─ engine/
│  ├─ retrieval/
│  ├─ clustering/
│  └─ utils/
├─ tests/
├─ data_qwen_split/
└─ PRD_跨应用移动界面语义相似度建模与场景聚类系统.md
```

## 7. 本期最小闭环

本期只要求跑通以下闭环：

1. 图像单模态分类
2. 多模态分类
3. embedding 提取
4. 检索示例
5. 聚类示例

做到这一步，已经足以支撑大创中期或结题答辩。

