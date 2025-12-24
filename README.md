This repository reproduces the **grokking phenomenon** described in
**Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"**
([arXiv:2201.02177](https://arxiv.org/abs/2201.02177))
using **modular division modulo 97**.

[中文版本](./README_CN.md)

The repo includes:

* training and validation accuracy/loss curves
* long-horizon grokking dynamics

---

## Overview

We study the task

\[
c = x \cdot y^{-1} \pmod{97}, \quad x \in \{0,\dots,96\},\; y \in \{1,\dots,96\}
\]

Each example is represented as a token sequence:

```
[x, ÷, y, =, c]
```

The model is trained in a **decoder-only, causal language modeling setup**, with
**loss and accuracy computed only on the answer token**.

The key phenomenon of interest is **grokking**:

* training accuracy reaches ~100% early,
* validation accuracy remains near chance for a long time,
* then suddenly rises in a sharp transition after many more optimization steps.

---

## Model and Training Details (Paper-Aligned)

The implementation follows Appendix A.1 of the paper:

* **Architecture**: decoder-only Transformer

  * 2 layers
  * width 128
  * 4 attention heads
  * causal self-attention
* **Objective**: cross-entropy loss on the answer token only
* **Optimizer**:

  * Adam (no weight decay) for late-grokking runs
  * AdamW (weight decay = 1) for faster, data-efficient variants
* **Learning rate**: 1e-3
* **Warmup**: 10 steps
* **Batch size**: `min(512, |train| / 2)` (as in the paper)
* **Training budget**:

  * up to **1e6 steps** for long-horizon grokking experiments

---

## Setup

### Requirements

* Python **3.13**
* PyTorch (CPU, CUDA, or MPS build as appropriate)

### Install

```bash
uv sync
```

---

## Training

### Canonical Grokking Reproduction (Paper-Style)

This is the **recommended setting** if your goal is to reproduce the qualitative
behavior shown in the paper (delayed generalization with a sharp transition).

```bash
uv run python main.py train \
  --p 97 \
  --train-frac 0.5 \
  --steps 1000000 \
  --eval-every 1000
```

**What to expect:**

* Training accuracy reaches ~1.0 very early.
* Validation accuracy stays near chance (~1/97) for a long time.
* After tens or hundreds of thousands of steps (seed-dependent),
  validation accuracy rises rapidly.
* Embeddings gradually develop a strong periodic/Fourier-like structure.

Outputs are written to:

```
runs/division_mod_97/
``` (or custom --out-dir)

---

### Quick Start

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

## Outputs

After training, the following files are produced under `runs/division_mod_97/` (or custom --out-dir):

### Logs

* `metrics.jsonl`
  Training and validation loss/accuracy over time.

---

## Visualization

### Training Curves

```bash
uv run python main.py plot --run-dir runs/division_mod_97
```

Generates:

* `plots/accuracy.png`
* `plots/loss.png`

---

## Reference

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V.
**Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**
arXiv:2201.02177

---

[English](./README.md) | [简体中文](./README_CN.md)
