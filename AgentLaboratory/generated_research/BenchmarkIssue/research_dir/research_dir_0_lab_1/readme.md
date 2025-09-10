# Advancing Symbolic Pattern Recognition through Hierarchical Reasoning

This repository contains the code, experiments, and documentation for our research on advancing symbolic pattern recognition (SPR) by leveraging hierarchical symbolic reasoning and hyperbolic embeddings. The framework focuses on two main innovations:

- **Discretisation of continuous latent spaces:** Converting continuous representations into a finite set of discrete symbols with vector quantisation (VQ).
- **Hierarchical reasoning in hyperbolic space:** Constructing an abstraction tree that captures exponential hierarchical dependencies via hyperbolic embeddings and operations.

The project serves as a baseline exploration—as demonstrated using a logistic regression classifier on SPR benchmarks—and sets the stage for future work integrating more sophisticated neuro-symbolic architectures.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Experiments and Results](#experiments-and-results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

Symbolic pattern recognition (SPR) is key to extracting hidden symbolic rules in complex datasets from domains such as natural language processing and visual scene interpretation. However, conventional models (e.g., linear classifiers) fall short in capturing the non-linear, hierarchical, and abstract symbolic dependencies.

Our approach addresses these limitations by:
- **Discretising latent representations:** Applying vector quantisation to map latent features into a discrete codebook.
- **Exploiting hyperbolic geometry:** Embedding the learned symbols in hyperbolic space to naturally account for tree-like, exponential growth of abstract features.
- **Hierarchical abstraction:** Building an abstraction tree using a multi-level reasoning module to extract interpretable symbolic rules.

The repository includes:
- A baseline logistic regression implementation using a revised token pattern for CountVectorizer.
- Detailed experimental settings and evaluations on four SPR benchmark datasets (SFRFG, IJSJF, GURSG, TSHUY).
- Code snippets for hyperbolic transformations and vector quantisation modules.

---

## Background

This work formulates the SPR task as a dual-stage process:
1. **Latent Extraction and Discretisation:**  
   Given an input instance x, a latent extractor f(x) produces a continuous representation z ∈ ℝᵈ. Next, z is discretised by mapping to the closest code vector from the codebook C using:
   
      min₍c ∈ C₎ ||z − c||₂
   
2. **Hierarchical Abstraction in Hyperbolic Space:**  
   To capture hierarchical relationships, a hyperbolic linear transformation is applied via:
   
      h(x) = (W ⊗₁ x) ⊕₁ B
   
   and the hyperbolic distance metric is defined as:
   
      d_H(u, v) = cosh⁻¹(⟨u, v⟩)
   
   These steps enable the formation of an abstraction tree that organizes symbols into hierarchies, reflecting the intrinsic structure of the data.

---

## Repository Structure

```
├── README.md
├── LICENSE
├── docs/
│   └── paper.pdf              # Full paper writeup (LaTeX version)
├── experiments/
│   ├── dataset_info.md        # Details on SPR benchmark datasets (SFRFG, IJSJF, GURSG, TSHUY)
│   ├── results.md             # Experimental results and analysis
│   └── run_experiments.sh     # Script to run baseline experiments
├── src/
│   ├── data_preprocessing.py  # Data collection, cleaning, and vectorization using CountVectorizer with revised token pattern
│   ├── vq_module.py           # Implementation of vector quantisation module
│   ├── hyperbolic_ops.py      # Hyperbolic transformations (Möbius scalar multiplication and addition)
│   ├── classifier.py          # Baseline logistic regression classifier integration
│   └── abstraction_tree.py    # Module to construct the hierarchical abstraction tree in hyperbolic space
└── requirements.txt           # List of required Python libraries (e.g., scikit-learn, numpy, matplotlib)
```

---

## Usage

### Prerequisites

- Python 3.7 or higher
- Recommended: Create a virtual environment

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Baseline Experiments

A sample end-to-end experiment pipeline is provided. To run the baseline logistic regression on the provided SPR benchmark datasets:

```bash
bash experiments/run_experiments.sh
```

Alternatively, you can run the main classifier script directly:

```bash
python src/classifier.py --dataset SFRFG
```

Parameters such as tokenization settings, regularization strength, and maximum iterations (set to 1000 by default) can be adjusted via command-line arguments or by modifying the configuration files.

---

## Experiments and Results

The baseline logistic regression model with the revised tokenization strategy achieves test accuracies of:

| Dataset | Test Accuracy (%) | SOTA Baseline (%) |
|---------|-------------------|-------------------|
| SFRFG   | 57.6              | 85.0              |
| IJSJF   | 58.5              | 80.0              |
| GURSG   | 58.5              | 83.0              |
| TSHUY   | 59.4              | 82.0              |

These results—documented in `experiments/results.md`—highlight the performance gap when using simple linear models for SPR tasks. The provided experiments include ablation studies (e.g., testing the effect of the revised token pattern) and statistical significance analysis.

The repository also contains scripts and visualizations demonstrating:
- The performance gap using bar charts (see figures in the LaTeX paper).
- How discretisation and hierarchical reasoning modules can be improved in future iterations.

---

## Future Work

Based on our findings, future developments will focus on:
- **Integration of non-linear and hierarchical models:** Incorporating transformer-based or recurrent architectures for enhanced symbolic reasoning.
- **End-to-end neuro-symbolic architectures:** Joint optimization of the VQ module and hyperbolic abstraction tree for improved symbolic rule extraction.
- **Refined discretisation techniques:** Exploring soft discretisations and adaptive loss functions.
- **Expanding evaluation metrics:** Beyond accuracy, assessing rule interpretability, robustness, and generalization performance.

Contributions and suggestions for improvements are very welcome!

---

## References

- arXiv:2503.04900v1 – Self-supervised symbolic sequence learning
- arXiv:2203.00162v3 – Evaluation of symbolic capabilities in Transformer architectures
- arXiv:1710.00077v1 – Pattern matching and symbolic computation techniques

For detailed theoretical background, please refer to the full paper available in the `docs/` folder.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy experimenting and thank you for your interest in advancing symbolic pattern recognition!