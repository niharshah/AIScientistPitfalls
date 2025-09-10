# Graph-Enhanced Differentiable Logic for SPR

This repository contains the code and supplementary materials for the paper:

**Research Report: Graph-Enhanced Differentiable Logic for SPR**  
*Agent Laboratory*

The paper introduces a novel graph-enhanced differentiable logic framework for symbolic pattern recognition (SPR). Our approach integrates Transformer-based embeddings, Graph Attention Networks (GATs), and differentiable logical reasoning modules—further refined via reinforcement learning (RL) based rule prototype generation—to extract interpretable symbolic rules from complex token sequences.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data and Experiments](#data-and-experiments)
- [Ablation Study](#ablation-study)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Symbolic pattern recognition (SPR) is a challenging task that requires accurate classification while retaining interpretability. This repository implements a framework that:

- **Encodes input sequences** using a Transformer encoder to capture both local and contextual dependencies.
- **Constructs a relational graph** from token embeddings based on positional proximity and feature similarity.
- **Refines intermediate representations** using a Graph Attention Network (GAT) to model non-local dependencies.
- **Projects the graph features** into soft predicate scores through differentiable logical operators like:
  - AND:  
    `AND(z) = ∏(z_i) for i = 1 to n`
  - OR:  
    `OR(z) = σ(∑(z_i))`  
    where σ denotes the sigmoid function.
- **Uses auxiliary losses**—logic fidelity loss, RL-based loss (via policy gradients), and supervised contrastive loss—to enforce crisp, symbolic interpretations.

Experimental validation on the SPR_BENCH dataset demonstrates competitive performance even under short (one-epoch) training regimes.

---

## Model Architecture

The overall framework consists of four primary components:

1. **Transformer Encoding and Graph Construction**  
   Each token in an L-token input sequence is mapped into a d-dimensional embedding space. A Transformer encoder captures the sequential as well as contextual features. Subsequently, a graph G=(V,E) is built where:
   - **Vertices (V):** Represent token embeddings.
   - **Edges (E):** Defined based on positional proximity and cosine similarity of features.

2. **Graph Attention and Logical Projection**  
   The constructed graph is processed by a GAT which computes attention weights (α₍ᵢⱼ₎) for node pairs using:
   ```
   α₍ᵢⱼ₎ = exp(LeakyReLU(aᵀ [W hᵢ ‖ W hⱼ])) / ∑ₖ exp(LeakyReLU(aᵀ [W hᵢ ‖ W hₖ]))
   ```
   Refined graph features are then projected into soft predicate scores using differentiable logical operators:
   - AND: z = ∏₍ᵢ=1₎ⁿ zᵢ
   - OR:  z = σ(∑₍ᵢ=1₎ⁿ zᵢ)

3. **Auxiliary Loss Functions**  
   Several loss components regulate training:
   - **Logic Fidelity Loss:**  
     ```
     L_logic = (1/N) ∑₍ᵢ=1₎ᴺ (zᵢ - round(zᵢ))²
     ```
   - **Reinforcement Learning Loss:** Utilizes policy gradients to refine rule prototypes.
   - **Supervised Contrastive Loss:** Encourages clustering of similar symbolic representations.
   The overall training objective is:
   ```
   L_total = L_cls + λ₁ L_logic + λ₂ L_RL + λ₃ L_supcon
   ```
   where L_cls is the primary classification loss.

4. **RL-based Rule Prototype Generator and Decision Module**  
   A reinforcement learning module generates candidate rule prototypes and a shallow MLP fuses outputs from the differentiable logic layer and RL module to output a final binary decision.

---

## Features

- **Neuro-Symbolic Integration:** Combines deep learning with symbolic reasoning.
- **Graph-Based Modeling:** Uses GAT for capturing non-local dependencies among tokens.
- **Differentiable Logic:** Implements logic operators in a fully differentiable manner.
- **Reinforcement Learning:** RL-based rule generation refines symbolic rule extraction.
- **Interpretable Outputs:** Soft predicate scores provide insights into the internal decision-making.

---

## Installation

Clone the repository using:

```bash
git clone https://github.com/yourusername/graph-enhanced-differentiable-logic-spr.git
cd graph-enhanced-differentiable-logic-spr
```

Install the required dependencies. It is recommended to use a virtual environment. For example, using pip:

```bash
pip install -r requirements.txt
```

*Dependencies include: PyTorch, NumPy, scikit-learn, and any additional packages specified in `requirements.txt`.*

---

## Usage

### Training

To train the model on the SPR_BENCH dataset:

```bash
python train.py --data_path path/to/spr_bench_dataset --epochs 1
```

*Note: In our experiments, we used a single epoch for demonstration. Extended training with hyperparameter tuning is encouraged for improved results.*

### Evaluation

To evaluate the trained model:

```bash
python evaluate.py --model_path path/to/saved_model --data_path path/to/spr_bench_dataset
```

Evaluation metrics include:
- Overall Accuracy
- Color-Weighted Accuracy (CWA)
- Shape-Weighted Accuracy (SWA)

---

## Data and Experiments

The SPR_BENCH dataset consists of synthetically generated sequences where each token is a combination of:
- 4 geometric shapes (△, □, ⊙, ◇)
- 4 colors (r, g, b, y)

The dataset is partitioned as follows:
- 1,000 training samples
- 300 development samples
- 500 testing samples

Experimental Setup Highlights:
- **Embedding Dimension:** 32
- **Transformer Encoder:** 1-layer
- **Graph Attention Network:** 4 attention heads
- **Learning Rate:** 1e-3
- **Batch Size:** 32

---

## Ablation Study

An ablation study is included in the codebase. The study demonstrates the impact of removing the GAT component (i.e., replacing it with simple average pooling) on performance metrics:

- With GAT:  
  - Test Accuracy: 60.60%
  - CWA: 61.37%
  - SWA: 60.25%

- Without GAT (average pooling):  
  - Accuracy: 48.67%
  - CWA: 47.78%
  - SWA: 46.37%

These results emphasize the role of graph-based relational modeling in capturing non-local dependencies.

---

## Future Work

Future explorations may include:
- **Extended Training:** Longer training epochs and refined hyperparameters to further close the gap to state-of-the-art performance (≈65% overall accuracy).
- **Enhanced Graph Strategies:** Investigating multi-scale graph representations and incorporating external semantic knowledge.
- **Improved Rule Generator:** Refining the reinforcement learning module for more precise symbolic rule extraction.
- **Visualization Tools:** Developing detailed visualizations for token-level attention, soft predicate activations, and rule generation analyses.

---

## License

This project is released under the [MIT License](LICENSE).

---

Feel free to open issues or submit pull requests for improvements, bug fixes, or future enhancements.

Happy coding and research!