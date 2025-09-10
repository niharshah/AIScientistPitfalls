# Hybrid Neuro-Symbolic SPR Detector

This repository contains the complete implementation, experimental results, and documentation for the "Hybrid Neuro-Symbolic SPR Detector for Synthetic PolyRule Reasoning." The project integrates a transformer-based encoder with a differentiable symbolic verifier into an end-to-end trainable architecture aimed at detecting adherence to a hidden rule defined over synthetic token sequences. The system is designed to bridge neural feature extraction and symbolic reasoning, ultimately providing interpretability alongside competitive performance.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Experiments & Results](#experiments--results)
- [Ablation Studies](#ablation-studies)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

The Synthetic PolyRule Reasoning (SPR) task requires deciding whether an input sequence of tokens—which each encode a shape and a color—satisfies a hidden rule composed as a conjunction of atomic predicates (e.g., shape-count, color-position, parity, and order constraints). Our hybrid model leverages:

- **Neural Module:** A transformer-based encoder to extract contextualized latent embeddings from the input sequence.
- **Symbolic Module:** A differentiable symbolic verifier that aggregates candidate rule scores and computes a final binary decision.

The overall objective is to maximize the likelihood:

  L(H, θ) = ∏₍S,y₎∈D Σ₍z∈Z₎ P(y, z | B, S, H, θ),

where H denotes the candidate symbolic rule pool, θ are the neural parameters, Z is the latent space, and B is the background knowledge.

---

## Architecture

The repository implements a hybrid neuro-symbolic framework with the following key components:

- **Transformer Encoder (φ):**  
  - Maps an input sequence S = [s₁, s₂, …, s_L] into a latent embedding z.
  - Uses 2 transformer layers with 4 attention heads and 32-dimensional token embeddings.

- **Candidate Generation Module (f):**  
  - A linear projection that maps the latent representation to a fixed candidate rule pool of size 4.

- **Differentiable Symbolic Verifier (g):**  
  - Aggregates the candidate rule scores to produce the final classification output.

- **Loss Function:**  
  - Combines binary cross-entropy (BCE) and a regularization term on the candidate rule distribution:
  
  L_total = L_BCE(ŷ, y) + λ R(f(φ(S)))

This design balances the robustness of neural feature extraction with the precision of symbolic rule verification.

---

## Installation

Clone the repository and install the required dependencies. The code has been developed and tested with Python 3.8+.

```bash
git clone https://github.com/your-username/hybrid-neuro-symbolic-spr.git
cd hybrid-neuro-symbolic-spr
pip install -r requirements.txt
```

*Requirements typically include PyTorch, NumPy, and other necessary libraries as detailed in `requirements.txt`.*

---

## Usage

To train and evaluate the model, execute the main training script. Ensure that the synthetic dataset is available in the `data/` directory (or adjust the dataset path accordingly).

```bash
python train.py --data_dir data/ --epochs 2 --batch_size 64 --lr 0.001
```

- **Training:**
  - The model is trained using the Adam optimizer.
  - Hyperparameters are set with a transformer embedding dimension of 32, 2 transformer layers, 4 attention heads, and a hidden dimension of 64 for the verifier.

- **Evaluation:**
  - Reports overall accuracy, Color-Weighted Accuracy (CWA), and Shape-Weighted Accuracy (SWA).
  - For example, in our experiments:
    - Training accuracy improved from 51.10% to 65.20%.
    - Development accuracy reached 77.80%.
    - Test set performance: 61.00% overall, 61.59% CWA, and 57.95% SWA.

---

## Dataset

The synthetic dataset is constructed with sequences of tokens, where each token consists of:
- A shape selected from {△, □, ○, ◊}
- A color selected from {r, g, b, y}

Each sequence is labeled to indicate if it satisfies a hidden rule defined by a conjunction of predicates (i.e., shape-count, color-position, parity, order constraints). The repository includes scripts to generate the synthetic data within the `data_generation/` folder.

---

## Experiments & Results

### Experimental Setup

- **Splits:** Training, development, and test sets with ~1000 examples each.
- **Metrics:** 
  - Overall Accuracy, 
  - Color-Weighted Accuracy (CWA), 
  - Shape-Weighted Accuracy (SWA)

### Main Results

| Metric           | Overall Accuracy (%) | CWA (%) | SWA (%) |
|------------------|----------------------|---------|---------|
| Our Model        | 61.00                | 61.59   | 57.95   |
| SOTA Baseline    | -                    | 65.00   | 70.00   |

*Note: While our model robustly captures color diversity, improvements are needed to handle complex shape arrangements.*

The repository also contains plotting scripts to visualize training loss and development accuracy over epochs (`plots/`).

---

## Ablation Studies

Ablation studies reveal:
- **Removing the Differentiable Symbolic Verifier:**  
  - Leads to a significant drop (~6–8%) in overall accuracy.
- **Candidate Pool Size Variation:**  
  - A fixed size of 4 is computationally efficient but may limit performance, particularly for shape complexities.
  
Additional details and evaluation metrics are provided in the associated experimental notebooks.

---

## Future Work

Potential directions for future improvements include:
- **Dynamic Candidate Rule Generation:**  
  - Allow the candidate pool to adapt based on input complexity.
- **External Memory Module Integration:**  
  - Enhance variable binding and sequential dependency tracking.
- **Advanced Hyperparameter Tuning & Regularization:**  
  - Experiment with deeper transformer configurations, alternative activation functions, and enhanced loss regularization measures.
- **Adaptive Error Correction Framework:**  
  - Incorporate algorithms to detect and correct systematic misclassifications.

---

## References

- NeuralFastLAS (arXiv:2310.05145v1)
- Neuro-symbolic Explanation Modules (arXiv:2112.12641v2)
- Chain-of-Thought Reasoning (arXiv:2212.08686v2)

For further details, please refer to the research paper provided in the repository.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or contributions, please open an issue or contact the maintainer at [your.email@domain.com].

Happy coding and experimenting with neuro-symbolic reasoning!