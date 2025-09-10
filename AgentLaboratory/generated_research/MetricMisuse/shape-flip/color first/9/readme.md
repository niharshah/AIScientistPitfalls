# Neuro-Symbolic Pattern Recognition

This repository contains the code, experiments, and detailed report for the project “Symbolic Integration of Neural and Symbolic Modules for Pattern Recognition.” The project focuses on the integration of transformer-based neural representations with an explicit, differentiable symbolic reasoning module to perform synthetic poly‐rule pattern recognition. While traditional neural models achieve competitive raw accuracy, our approach leverages explicit atomic predicate outputs to yield improved interpretability, albeit with initial challenges in predictive performance.

---

## Table of Contents

- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion & Future Work](#discussion--future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

In many pattern recognition tasks, particularly those involving synthetic and combinatorial rules, the trade-off between performance and model interpretability becomes critical. Our project addresses this issue by proposing a Symbolic Rule Network (SRN). The SRN integrates:

- **Transformer-based Neural Representations**: Capturing contextual dependencies among input tokens.
- **Differentiable Symbolic Reasoning Module**: Generating explicit atomic predicate outputs (e.g., shape-count, color-position, parity, order) which are aggregated via a differentiable logical AND.

The research shows that while the baseline transformer classifier achieves an overall accuracy of 62.0% (with Color-Weighted Accuracy of 63.03% and Shape-Weighted Accuracy of 61.45%), the initial SRN—implemented with a multiplicative aggregation mechanism—faces challenges, achieving only around 30.0% overall accuracy. Nonetheless, the explicit reasoning processes of the SRN provide a promising direction for scenarios where interpretability is paramount.

---

## Project Motivation

The core motivation behind the project is to overcome the “black-box” nature of deep learning models by:
- Integrating symbolic reasoning directly into the neural pipeline.
- Decomposing complex decision rules into interpretable atomic predicates.
- Providing a framework for future improvements in aggregation mechanisms to bridge the gap between interpretability and predictive performance.

This GitHub repository aggregates our research paper, source code, experimental data, and notes related to symbolic integration in neuro-symbolic systems.

---

## Repository Structure

Below is a brief description of the repository organization:

```
.
├── data/                        # Synthetic dataset files and preprocessed data
├── docs/                        # Additional documentation and extended reports (including the research paper)
├── experiments/                 # Scripts for running experiments, ablation studies, and hyperparameter tuning
├── models/                      # Implementation of the Transformer baseline and SRN architectures
│   ├── baseline_transformer.py  # Baseline transformer classifier code
│   └── symbolic_rule_network.py # SRN with differentiable symbolic reasoning module
├── utils/                       # Utility functions for tokenization, evaluation metrics (Accuracy, CWA, SWA), etc.
├── figures/                     # Figures used in the report (e.g., model architecture, aggregation mechanism comparisons)
├── requirements.txt             # Python package dependencies
└── README.md                    # This file
```

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/neuro-symbolic-pattern-recognition.git
   cd neuro-symbolic-pattern-recognition
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Baseline Transformer

To train and evaluate the baseline transformer model, run:

```bash
python experiments/run_baseline.py --config experiments/configs/baseline_config.yaml
```

### Running the Symbolic Rule Network (SRN)

For running experiments with the SRN, use:

```bash
python experiments/run_srn.py --config experiments/configs/srn_config.yaml
```

### Evaluation Metrics

The evaluation scripts compute:
- **Overall Accuracy**: Percentage of correctly classified instances.
- **Color-Weighted Accuracy (CWA)**: Accuracy weighted by the diversity of colors in input sequences.
- **Shape-Weighted Accuracy (SWA)**: Accuracy weighted by the diversity of shapes.

The metrics and evaluation logic are implemented in the `utils/metrics.py` file.

---

## Experimental Setup

- **Dataset**: We use a synthetic poly‐rule pattern recognition (SPR) dataset where each instance is a sequence of tokens denoting shapes (e.g., △, □, ○, ♦) and colors (e.g., r, g, b, y). The dataset is split into training, development, and testing subsets.
- **Model Architecture**:
  - **Transformer Encoder**: Processes token embeddings (with separate embedding layers for shapes and colors) combined with positional encoding.
  - **Predicate Heads**: For each atomic predicate (shape-count, color-position, parity, order), a dedicated linear layer computes its probability using a sigmoid activation.
  - **Aggregation**: Predicate probabilities are aggregated via a product operation (as a differentiable logical AND) and complemented by an aggregate prediction head.
  
- **Training Details**: 
  - Optimizer: Adam (learning rate = 1e-3)
  - Loss Function: Binary Cross-Entropy (BCE)
  - Training Epochs: Initially one epoch on reduced dataset subsets (for rapid prototyping). Extended training on full datasets is planned for future experiments.

Scripts and configuration details are provided in the `experiments/` directory.

---

## Results

### Baseline Transformer Results

- **Overall Accuracy**: 62.0%
- **Color-Weighted Accuracy (CWA)**: 63.03%
- **Shape-Weighted Accuracy (SWA)**: 61.45%

### Symbolic Rule Network (SRN) Results

- **Overall Accuracy**: 30.0%
- **Color-Weighted Accuracy (CWA)**: 29.70%
- **Shape-Weighted Accuracy (SWA)**: 28.49%

The SRN’s lower performance is primarily attributed to the sensitivity of the multiplicative aggregation mechanism employed to combine predicate outputs. Detailed experimental results, comparisons with state-of-the-art (SOTA) benchmarks, and ablation studies are discussed in the report (see `docs/research_report.pdf`).

---

## Discussion & Future Work

The report analyzes:
- The trade-off between interpretability and performance in neuro-symbolic reasoning.
- The brittleness of the product-based aggregation mechanism that amplifies individual predicate errors.
- Future improvements including:
  - Alternative soft logical operators (e.g., weighted sums or learned aggregation functions).
  - Extended training regimes and hyperparameter tuning.
  - Architectural refinements for the symbolic module (e.g., using recurrent or graph neural networks).
  - Integration of external symbolic knowledge.

The full discussion, error analyses, and future research directions are provided in the research report available under `docs/research_report.pdf`.

---

## Citation

If you find this work useful for your research, please cite our report:

    @misc{neuro-symbolic2023,
      title={Symbolic Integration of Neural and Symbolic Modules for Pattern Recognition},
      author={Agent Laboratory},
      year={2023},
      note={Available at https://github.com/yourusername/neuro-symbolic-pattern-recognition},
    }

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

We welcome contributions, discussions, and feedback. If you have any questions or suggestions, please feel free to open an issue or submit a pull request.

Happy coding!