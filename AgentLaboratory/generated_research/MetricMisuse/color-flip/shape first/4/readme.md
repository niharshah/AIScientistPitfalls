# Neuro-Symbolic Hybrid Integration for the SPR Task

Welcome to the Neuro-Symbolic Hybrid Integration repository! This project presents a novel approach for tackling the Sequence Pattern Recognition (SPR) task by seamlessly combining deep learning representations with explicit symbolic reasoning. The approach leverages a lightweight Transformer encoder along with differentiable predicate extraction modules (shape-count, color-position, parity, and order) to achieve interpretable decision-making.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results & Discussion](#results--discussion)
- [Citation](#citation)
- [License](#license)

---

## Overview

The SPR (Sequence Pattern Recognition) task involves determining whether an input token sequence satisfies a hidden composite rule. Each token encodes an abstract symbol defined by a shape (from {△, ■, •, ◇}) and a color (from {r, g, b, y}). Our approach embeds the tokens, sums their learned representations, and processes them with a Transformer encoder. The resulting signal is then passed through four differentiable predicate modules whose activations are aggregated using a differentiable logical AND (implemented as a product operation).

This repository contains:

- The research paper detailing the method and experiments.
- Code implementing the neuro-symbolic hybrid model as well as an ablation baseline.
- Scripts and notebooks for data preprocessing, training, and evaluation on the SPR_BENCH dataset.
- Analysis artifacts such as training curves and predicate activation histograms.

---

## Features

- **Neuro-Symbolic Integration:** Combines continuous deep learning representations with discrete, interpretable symbolic predicates.
- **Transformer-based Encoder:** Processes token sequences to capture global and local dependencies.
- **Explicit Predicate Modules:** Four symbolic predicate modules (shape-count, color-position, parity, and order) that contribute interpretable activations.
- **Differentiable Logical Aggregation:** Uses a product operation (differentiable logical AND) to fuse predicate activations into a final decision.
- **Comparative Analysis:** Includes an ablation model that bypasses explicit predicate extraction for performance comparison.
- **Reproducible Research:** Fixed random seed and controlled dataset splits ensure reproducibility.

---

## Repository Structure

The repository is organized as follows:

```
.
├── data/
│   └── SPR_BENCH/                   # SPR_BENCH dataset files (or instructions to download)
├── notebooks/
│   └── exploratory_analysis.ipynb   # Notebook for activation histograms and training curves
├── src/
│   ├── models.py                    # Implementation of Hybrid and Ablation models
│   ├── train.py                     # Training script for model training
│   ├── evaluate.py                  # Evaluation script to compute Shape-Weighted Accuracy (SWA)
│   └── utils.py                     # Utility functions for preprocessing and visualization
├── experiments/
│   └── hyperparameters.md           # Summary of key hyperparameters and training settings
├── paper/
│   └── NeuroSymbolic_SPR.pdf        # The full research paper
└── README.md                        # This file
```

---

## Installation

### Requirements

- Python 3.7+
- PyTorch (tested with v1.9 or later)
- Other dependencies: NumPy, SciPy, Matplotlib, and scikit-learn

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/neuro-symbolic-spr.git
   cd neuro-symbolic-spr
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Download the SPR_BENCH dataset:**

   Follow the instructions in the `data/SPR_BENCH/README.md` file to obtain and prepare the dataset.

---

## Usage

### Training the Model

To train the neuro-symbolic hybrid model, run:

```bash
python src/train.py --model hybrid --data_dir data/SPR_BENCH --epochs 1
```

For training the ablation model (without explicit predicate extraction), run:

```bash
python src/train.py --model ablation --data_dir data/SPR_BENCH --epochs 1
```

_Parameters such as learning rate, batch size, and embedding dimensions are set according to the hyperparameters detailed in [Experimental Setup](#experimental-setup)._

### Evaluating the Model

After training, evaluate the model using:

```bash
python src/evaluate.py --model path/to/trained/model.pt --data_dir data/SPR_BENCH
```

This script computes the Shape-Weighted Accuracy (SWA) metric and generates visualizations (e.g., predicate activation histograms).

### Notebooks

The `notebooks/exploratory_analysis.ipynb` notebook provides additional details on data preprocessing, training curves, and the analysis of predicate activations. It is a good starting point if you want to interactively explore the performance and interpretability aspects.

---

## Experimental Setup

Our experiments were conducted on the SPR_BENCH dataset, which contains synthetically generated token sequences. Key settings include:

- **Data Splits:** 500 training samples, 100 development samples, and 100 test samples.
- **Embedding Dimension:** 16
- **Transformer Encoder:** 1 layer with 4 attention heads
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Decision Threshold (τ):** 0.5

The full experimental setup—including details on the data preprocessing pipeline, loss function (binary cross-entropy), and evaluation metric (Shape-Weighted Accuracy or SWA)—is described in the paper and the `experiments/hyperparameters.md` file.

---

## Results & Discussion

Our neuro-symbolic hybrid model achieved a test set SWA of 52.25%, while the ablation model (which omits explicit predicate extraction) reached 61.24% SWA. Although the ablation model boasts higher raw accuracy, the hybrid approach provides valuable interpretability by exposing the individual predicate contributions (e.g., shape-count, color-position, parity, and order) through activation histograms.

The discussion in the paper elaborates on the trade-offs between interpretability and performance, the implications of our differentiable logical AND aggregation, and future research directions—such as incorporating additional symbolic modules and refining the predicate integrations to narrow the performance gap.

For a detailed analysis, please refer to the `paper/NeuroSymbolic_SPR.pdf` document.

---

## Citation

If you use this work in your research, please consider citing our paper:

```
@article{neurosymbolic_spr2023,
  title={Neuro-Symbolic Hybrid Integration for the SPR Task},
  author={Agent Laboratory},
  journal={Preprint},
  year={2023},
  note={Available on arXiv and GitHub repository: https://github.com/yourusername/neuro-symbolic-spr}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to open issues or submit pull requests if you have any questions or improvements. We welcome contributions to advance the integration of neuro-symbolic methods in transparent AI systems.

Happy coding!