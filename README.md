# Hybrid-GNN-Tree-Molecular-Predictor
Hybrid machine learning architecture fusing local 2D graph topology (GIN/GATv2) with global tabular features (XGBoost/RF/MLP) via a Logistic Regression meta-model for molecular property classification. Built by Team Data AImers for OpenAImers 2026 Track A.
# 🧬 Multi-Modal Molecular Stacking Pipeline (OpenAImers 2026 - Track A)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-ee4c2c.svg)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange.svg)

> **Team Data AImers** | High-accuracy classification system for predicting 5 binary molecular properties.

## 📌 Overview
Standard Graph Neural Networks (GNNs) often fail to capture global molecular macro-properties (like total weight or solubility), while traditional Tree-based models completely fail to understand 2D structural bond topology. 

To solve this, we built a **Multi-Modal Molecular Stacking Pipeline** from scratch (no massive pre-trained transformer weights). It separates local bond extraction from global property analysis and mathematically merges them using a highly stable rank-based Meta-Model. 

## 🏗️ High-Level Architecture

Our architecture features a two-level ensembling approach:

### Level 1: Multi-View Extraction & Base Engines
1. **Feature Extraction (`RDKit`):** Simultaneously extracts 2D graph topology (nodes/edges) and **2223** dense global tabular features (Morgan Fingerprints, MACCS Keys, RDKit Descriptors).
2. **Local Graph Engines (`PyTorch Geometric`):** Uses **GIN** and **GATv2** models equipped with **Jumping Knowledge** to concatenate representations across all message-passing layers, preventing over-smoothing on complex molecules.
3. **Global Tabular Engines (`Scikit-Learn` / `XGBoost`):** Processes the dense 2223-dimensional fingerprints using a trio of models: **XGBoost** (Gradient Boosting), **Random Forest** (Bagging), and a **Deep MLP** (Scaled Non-Linear). 

### Level 2: The Meta-Orchestrator
Instead of relying on a simple average, all Level-1 Out-Of-Fold (OOF) predictions are converted to uniform percentiles via **Rank Transformation (`scipy.stats.rankdata`)**. These are fed into a heavily regularized, class-balanced **Logistic Regression Meta-Model**, completely bypassing probability calibration clashes between Trees and GNNs.

## 🚀 Key Innovations & Edge Case Handling

* **Murcko Scaffold Split (8-Fold):** Divides the dataset based on core 2D ring structures rather than random sampling. This prevents data leakage and forces models to generalize to unseen chemical backbones, perfectly simulating a hidden Private Leaderboard.
* **Dummy Atom Fallback:** Invalid or broken SMILES strings are silently substituted with a single Carbon atom (`'C'`). This maintains perfect array dimensions and guarantees zero pipeline crashes during inference.
* **Extreme Target Imbalance Resolution:** Deploys **Focal Loss** (`gamma=2.0`) combined with dynamic positive class weights to force the neural networks to prioritize rare active molecules, directly maximizing the MCC metric.
* **Clamped MCC Optimization:** The threshold optimizer is strictly clamped between `0.45` and `0.75` to prevent the algorithm from aggressively chasing precision and triggering catastrophic False Negatives.

## ⚙️ Installation & Usage

### Requirements
* `torch` >= 2.0.0
* `torch_geometric`
* `rdkit`
* `xgboost`
* `scikit-learn`
* `scipy`
* `pandas`, `numpy`

### Running the Pipeline
Clone the repository and ensure your `train.csv` and `test.csv` are in the root directory.

```bash
git clone [https://github.com/yourusername/DataAImers-Molecular-Stacking.git](https://github.com/yourusername/DataAImers-Molecular-Stacking.git)
cd DataAImers-Molecular-Stacking
jupyter notebook data-aimers-track-a.ipynb
# Or upload directly to Kaggle/Google Colab and run all cells
