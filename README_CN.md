# Grokking 实验复现

本项目复现了 **Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic
Datasets"** ([arXiv:2201.02177](https://arxiv.org/abs/2201.02177)) 论文中描述的 **grokking 现象**，
使用 **模 97 除法** 作为实验任务。

本仓库包含：

* 训练和验证准确率/损失曲线
* 长程 grokking 动力学分析

---

## 概述

我们研究的任务是：

$$
c = x \cdot y^{-1} \pmod{97}, \quad x \in \{0,\dots,96\},\; y \in \{1,\dots,96\}
$$
$$

每个样本表示为一个 token 序列：

```
[x, ÷, y, =, c]
```

模型采用 **解码器-only、因果语言建模** 设置，**仅在答案 token 上计算损失和准确率**。

我们关注的 **grokking** 现象：

* 训练准确率早期就达到 ~100%
* 验证准确率在很长时间内保持接近随机水平
* 然后在大量优化步骤后突然上升，出现急剧转变

---

## 模型和训练细节（与论文一致）

实现遵循论文附录 A.1：

* **架构**：解码器-only Transformer

  * 2 层
  * 宽度 128
  * 4 个注意力头
  * 因果自注意力
* **目标**：仅在答案 token 上计算交叉熵损失
* **优化器**：

  * Adam（无权重衰减）用于延迟 grokking 实验
  * AdamW（权重衰减 = 1）用于更快、数据高效的变体
* **学习率**：1e-3
* **预热**：10 步
* **批量大小**：`min(512, |train| / 2)`（与论文一致）
* **训练预算**：

  * 长程 grokking 实验最多 **1e6 步**

---

## 环境配置

### 依赖

* Python **3.13**
* PyTorch（CPU、CUDA 或 MPS 版本，根据需要选择）

### 安装

```bash
uv sync
```

---

## 训练

### 标准 Grokking 复现（论文风格）

这是**推荐设置**，如果你想复现论文中展示的定性行为（延迟泛化伴随即急剧转变）。

```bash
uv run python main.py train \
  --p 97 \
  --train-frac 0.5 \
  --steps 1000000 \
  --eval-every 1000
```

**预期结果：**

* 训练准确率很早就达到 ~1.0
* 验证准确率在很长时间内保持接近随机（~1/97）
* 经过数万或数十万步后（取决于种子），验证准确率快速上升
* 嵌入逐渐形成强烈的周期/类傅里叶结构

输出写入：

```
runs/division_mod_97/
```

---

### 快速开始

```python
python3 main.py train \
  --preset paper_late \
  --p 97 \
  --train-frac 0.5 \
  --seed 0 \
  --eval-every 1000 \
  --out-dir runs/division_mod_97_pair_paper_late
```

---

## 输出

训练完成后，在 `runs/division_mod_97/`（或自定义 --out-dir）下生成以下文件：

### 日志

* `metrics.jsonl`
  训练和验证损失/准确率随时间的变化。

---

## 可视化

### 训练曲线

```bash
uv run python main.py plot --run-dir runs/division_mod_97
```

生成：

* `plots/accuracy.png`
* `plots/loss.png`

---

## 参考

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V.
**Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**
arXiv:2201.02177

---

[English](./README.md) | 简体中文
