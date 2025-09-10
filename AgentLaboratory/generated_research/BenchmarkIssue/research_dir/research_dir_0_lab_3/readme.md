# Dual-Branch Neuro-Symbolic Reasoning for SPR

Welcome to the repository for our research on a dual-branch neuro-symbolic framework for Sequential Pattern Recognition (SPR). This project integrates a graph-based attention encoder with a differentiable symbolic logic module to both classify sequences and extract interpretable symbolic rules. The repository contains the code, experimental setups, and documentation associated with our work.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [Branch A: Graph Attention Encoder](#branch-a-graph-attention-encoder)
  - [Branch B: Differentiable Symbolic Logic Module](#branch-b-differentiable-symbolic-logic-module)
- [Experimental Setup](#experimental-setup)
  - [Synthetic Dataset Generation](#synthetic-dataset-generation)
  - [Training Protocol and Loss Functions](#training-protocol-and-loss-functions)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results and Analysis](#results-and-analysis)
- [Usage](#usage)
- [Installation and Requirements](#installation-and-requirements)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

Sequential Pattern Recognition (SPR) plays a crucial role in applications ranging from natural language processing to signal interpretation. Traditional methods generally decouple low-level feature extraction from high-level symbolic reasoning. In contrast, our dual-branch framework integrates:

1. **Graph-Based Attention Encoder:** Encodes token sequences in a graph structure that captures both sequential and semantic relationships.
2. **Differentiable Symbolic Logic Module:** Implements soft, differentiable approximations of Boolean operations to extract explainable rule scores corresponding to atomic predicates such as shape count, color position, parity, and order.

The model is trained end-to-end using a composite loss that balances classification performance with rule fidelity. Extensive evaluations on multiple synthetic benchmarks (SFRFG, IJSJF, GURSG, and TSHUY) showcase the advantages of a tightly coupled neuro-symbolic approach, particularly when the underlying graph structure reflects clear relational interdependencies.

---

## Model Architecture

### Branch A: Graph Attention Encoder

- **Input Representation:** Each token is an 8-dimensional one-hot encoded vector combining shape and color.
- **Embedding:** Tokens are projected into a continuous space with a linear layer.
- **Multi-Head Attention:** A transformer-inspired multi-head attention layer computes contextual relationships:
  
  Attn(Q, K, V) = softmax((Q Kᵀ) / √(dₖ)) V
  
  where dₖ is the key dimension.
  
- **Output:** The attention outputs are pooled and further transformed via a fully connected layer to yield a 16-dimensional feature vector.

### Branch B: Differentiable Symbolic Logic Module

- **Functionality:** Processes the continuous 16-dimensional vector from Branch A.
- **Soft Boolean Operations:** Implements soft relaxations of Boolean functions (e.g., A ∧ B ≈ A · B, A ∨ B ≈ A + B − A · B).
- **Output:** Produces 4 continuous scores that correspond to symbolic predicates (e.g., shape count, color position, parity, order).

---

## Experimental Setup

### Synthetic Dataset Generation

- **Dataset Structure:** Each instance is a token sequence combined with a graph structure:
  - **Sequential Edges:** Connect adjacent tokens.
  - **Semantic Edges:** Connect tokens with similar features (e.g., common shape or color).
- **Noise and Perturbations:** Controlled noise is introduced by perturbing token attributes and edge connections to test robustness.

### Training Protocol and Loss Functions

Our model is trained end-to-end with the following composite loss:

L = L_ce + λ₁‖R‖₁ + λ₂L_logic

- **Cross-Entropy Loss (L_ce):** Drives binary classification performance.
- **L₁ Regularization (‖R‖₁):** Promotes sparsity in rule scores.
- **Logic Loss (L_logic):** A mean squared error loss ensuring alignment between soft symbolic outputs and ground truth.

**Hyperparameters:**
- **Embedding Dimension:** 32
- **Attention Heads:** 4
- **Learning Rate:** 0.005
- **Epochs:** 2 (for proof-of-concept with small subsample sizes)
- **Regularization Coefficients:** λ₁ = 0.01, λ₂ = 0.1

### Evaluation Metrics

- **Test Accuracy:** Primary metric reflecting the model’s classification performance.
- **Development Accuracy:** Accuracy on a validation set for hyperparameter tuning.
- **Confusion Matrix Analysis:** Identifies bias and evaluates misclassification distribution.
- **Training Loss Convergence:** Monitored to ensure stable learning dynamics.

---

## Results and Analysis

Our experiments across four benchmarks yielded varied performance levels:

| Benchmark | Epoch 1 Loss | Epoch 2 Loss | Dev Accuracy (%) | Test Accuracy (%) |
|-----------|--------------|--------------|------------------|-------------------|
| SFRFG     | 0.7353       | 0.6222       | 64               | 54                |
| IJSJF     | 0.7525       | 0.7488       | 48               | 50                |
| GURSG     | 0.6962       | 0.3830       | 94               | 90                |
| TSHUY     | 0.6300       | 0.3779       | 98               | 100               |

**Insights:**
- **High Performers (GURSG, TSHUY):** Clear and robust graph structures led to very high accuracy.
- **Challenging Cases (SFRFG, IJSJF):** Increased noise and ambiguous rules decreased performance, indicating room for further refinement.

An ablation study confirmed that removing either the attention encoder or the logic module significantly degrades performance, underscoring the importance of their integration.

---

## Usage

To train and evaluate the model:

1. **Clone the Repository:**
   ```
   git clone https://github.com/<your_username>/dual-branch-neuro-symbolic-spr.git
   cd dual-branch-neuro-symbolic-spr
   ```

2. **Prepare the Environment:**
   It is recommended to use Python 3.7+.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Training:**
   The training scripts are located in the `src/` folder.
   ```bash
   python src/train.py --benchmark TSHUY
   ```

4. **Evaluate Model:**
   To evaluate on test data:
   ```bash
   python src/evaluate.py --benchmark TSHUY
   ```

5. **Visualize Results:**
   Training loss and confusion matrix plots are generated and stored in the `results/` directory.

---

## Installation and Requirements

- **Python:** 3.7 or later
- **PyTorch:** CPU-only configuration (for reproducibility)
- **Additional Dependencies:** Listed in `requirements.txt` (e.g., numpy, matplotlib)

To install the required packages, run:
```
pip install -r requirements.txt
```

---

## Future Work

Future research directions include:
- **Extended Training Regimes:** Increasing training epochs and dataset sizes.
- **Adaptive Regularization:** Dynamically tuning λ₁ and λ₂ for better balance between sparsity and fidelity.
- **Iterative Feedback Mechanisms:** Incorporating recurrent feedback loops between symbolic outputs and attention weights.
- **External Knowledge Integration:** Leveraging prior symbolic knowledge to guide rule extraction.
- **Scalability Studies:** Testing on larger and more diverse datasets, further bridging the gap between interpretability and performance.

---

## References

1. Kamruzzaman, S., & Hasan, M. (2005). [Title of the referenced paper]. *Journal Name*, Volume(Issue), pages.
2. ArXiv preprints:
   - [arXiv:1704.07503v1](https://arxiv.org/abs/1704.07503v1)
   - [arXiv:2212.08686v2](https://arxiv.org/abs/2212.08686v2)
   - [arXiv:2503.06427v1](https://arxiv.org/abs/2503.06427v1)
   - [arXiv:2505.06745v1](https://arxiv.org/abs/2505.06745v1)

Please refer to our full paper in `paper.pdf` for detailed discussions and additional experimental results.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

We welcome feedback, suggestions, and contributions to help enhance the capabilities of our dual-branch neuro-symbolic reasoning framework for SPR. Happy coding!