# Edge-ANNoy: Enabling Efficient Light ANN Framework for Remote Sensing Image Retrieval on Storage-Constrained Edge Devices

[![GitHub Stars](https://img.shields.io/github/stars/YourOrg/Edge-ANNoy?style=social)](https://github.com/YourOrg/Edge-ANNoy/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/YourOrg/Edge-ANNoy?style=social)](https://github.com/YourOrg/Edge-ANNoy/network/members)
[![License](https://img.shields.io/github/license/YourOrg/Edge-ANNoy)](https://github.com/YourOrg/Edge-ANNoy/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
<!-- 您可以根据需要添加更多徽章，如 CI/CD 状态、文档状态等 -->

## 1. 项目简介

Edge-ANNoy 是一个创新的近似最近邻搜索（ANNS）框架，专为存储受限的边缘设备（如微型卫星和无人机）上的遥感图像检索而设计。针对传统基于树的 ANNoy 方法存储高维超平面所带来的存储挑战，Edge-ANNoy 核心创新在于其**不显式存储高维超平面**。相反，它利用数据集中现有的数据项对，即“锚点”（anchors），来**隐式定义空间分区**。为了确保这些分区既平衡又高效，我们开发了一种新颖的**二元锚点优化（Binary Anchor Optimization）算法**。

这一架构转变将模型的空间复杂度从依赖于数据维度 $d$ 的 $O(t_n(dN/T + N))$ 降低到**完全与特征维度解耦**的 $O(t_n(N/T + N))$。严格的实验表明，在模拟边缘双存储约束环境下，Edge-ANNoy 相比基线 ANNoy 在次级存储方面实现了 30-40% 的减少，且仅牺牲了微小的 3-5% 检索精度。其在受限场景下的整体检索性能超越了其他主流方法。Edge-ANNoy 有望成为在存储极度受限的边缘设备上实现大规模、高性能、实时遥感图像检索的领先解决方案。

## 2. 特性

*   **维度无关的轻量级架构**:
    *   提出了基于“锚点对 + 标量偏移”机制的隐式超平面表示。
    *   将模型空间复杂度从 $O(t_n(dN/T + N))$ 降低至 $O(t_n(N/T + N))$，彻底**解耦了模型大小与特征维度 $d$ 的关系**，对边缘设备的主次级存储约束至关重要。
*   **基于二元锚点优化的平衡分区机制**:
    *   引入了一种新颖的优化算法，以在树构建过程中改善分区平衡。
    *   通过选择大致平分数据子集的锚点对，并微调超平面偏移量，有效缓解了树不平衡导致的性能退化，平衡了检索效率和召回精度。
*   **索引结构存储**: 支持将构建好的树结构（节点信息和叶子节点数据ID）存储到文本文件中。
*   **经过全面边缘环境验证**: 在 GUN、Hi-UCD 和 MillionAID 三个大规模遥感数据集上，通过严格的资源预算实验，验证了 Edge-ANNoy 在存储效率和推理速度上的显著优势，同时保持了有竞争力的检索精度。

## 3. 安装

### 3.1 环境要求

*   Python 3.8 或更高版本
*   pip 
*   PyTorch 

### 3.2 依赖安装

建议为项目创建一个独立的 Python 虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# 安装项目依赖
pip install -r requirements.txt