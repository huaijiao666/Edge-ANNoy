# Edge-ANNoy: Enabling Efficient Light ANN Framework for Remote Sensing Image Retrieval on Storage-Constrained Edge Devices

[![GitHub Stars](https://img.shields.io/github/stars/huaijiao666/Edge-ANNoy?style=social)](https://github.com/huaijiao666/Edge-ANNoy/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/huaijiao666/Edge-ANNoy?style=social)](https://github.com/huaijiao666/Edge-ANNoy/network/members)
[![License](https://img.shields.io/github/license/huaijiao666/Edge-ANNoy)](https://github.com/huaijiao666/Edge-ANNoy/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

## 1. Project Introduction

Edge-ANNoy is an innovative Approximate Nearest Neighbor Search (ANNS) framework specifically engineered for remote sensing image retrieval on storage-constrained edge devices, such as micro-satellites and UAVs. Addressing the storage challenges posed by traditional tree-based ANNoy methods that explicitly store high-dimensional hyperplanes, Edge-ANNoy's core innovation lies in its **departure from storing explicit high-dimensional hyperplanes**. Instead, it leverages pairs of existing data items, termed "anchors," to **implicitly define spatial partitions**. To ensure these partitions are both balanced and effective, we have developed a novel **Binary Anchor Optimization algorithm**.

This architectural shift reduces the model’s space complexity from $O(t_n(dN/T + N))$, which is dependent on the data dimension $d$, to $O(t_n(N/T + N))$, thereby **completely decoupling it from the feature dimension**. Rigorous experiments demonstrate that, under simulated edge environments with dual storage constraints, Edge-ANNoy achieves a 30-40% reduction in secondary storage compared to the baseline ANNoy, at the cost of a minor 3-5% drop in retrieval accuracy. Furthermore, its overall retrieval performance surpasses that of other mainstream methods in these constrained scenarios. Collectively, these results establish Edge-ANNoy as a state-of-the-art solution for enabling large-scale, high-performance, real-time remote sensing image retrieval on edge devices with exceptionally constrained storage.

## 2. Features

*   **Dimension-Independent Lightweight Architecture**:
    *   Proposes an implicit hyperplane representation based on an "anchor pair + scalar offset" mechanism.
    *   Reduces the model's space complexity from $O(t_n(dN/T + N))$ to $O(t_n(N/T + N))$, significantly decreasing the model size and **decoupling it from the feature dimension $d$**, which is critical for adapting to the primary and secondary storage constraints of edge devices.
*   **Balanced Partitioning Mechanism via Binary Anchor Optimization**:
    *   Introduces a novel optimization algorithm to improve partition balance during tree construction.
    *   By selecting anchor pairs that approximately bisect the data subset and fine-tuning the hyperplane offset, our method mitigates the performance degradation typically induced by tree imbalance, effectively balancing retrieval efficiency and recall accuracy.
*   **Comprehensive Edge-Oriented Validation**: Through systematic experiments on the GUN, Hi-UCD, and MillionAID large-scale remote sensing datasets under tight resource budgets, Edge-ANNoy demonstrates significant gains in storage efficiency and inference speed while preserving competitive retrieval accuracy.

## 3. Installation

### 3.1 Environment Requirements

*   Python 3.8 or higher
*   pip (Python package installer)
*   PyTorch (It is highly recommended to install the CUDA-enabled version for optimal performance. Please refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) for installation guides.)

### 3.2 Dependency Installation

It is recommended to create an isolated Python virtual environment for the project:

```bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# Install project dependencies
pip install -r requirements.txt