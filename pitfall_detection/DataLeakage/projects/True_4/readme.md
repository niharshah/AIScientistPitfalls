# Hybrid Grid-CNN with Latent HMM for Symbolic Pattern Recognition

Welcome to the official GitHub repository for the project "Hybrid Grid-CNN with Latent HMM for Symbolic Pattern Recognition". This repository hosts the code, experiments, and supplementary materials described in our research paper. The project details a novel hybrid model that integrates a sequence-to-grid transformation, a convolutional neural network (CNN), and a latent Hidden Markov Model (HMM) module for Symbolic Pattern Recognition (SPR).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

Symbolic Pattern Recognition (SPR) involves determining whether an input sequence of abstract symbols satisfies an implicit and complex rule. In our approach, we transform the one-dimensional sequence into a structured two-dimensional grid, enabling a CNN to capture both local and global features. Alongside, a latent HMM module infers candidate latent rules and assigns probability scores to each candidate, ensuring interpretability and richer rule extraction. The entire architecture is trained end-to-end on a synthetic SPR benchmark dataset using binary cross-entropy loss optimized with the Adam optimizer.

**Key Contributions:**
- Novel sequence-to-grid transformation that facilitates effective CNN-based spatial feature extraction.
- Integration of a latent HMM module that infers candidate latent predicates, enforcing a probability constraint.
- Comprehensive experimental evaluation of different latent candidate configurations (e.g., C=4 and C=8) to study trade-offs between interpretability and classification performance.

---

## Project Structure

Below is an overview of the repository structure:

```
├── data/                       # Synthetic SPR dataset and preprocessing scripts
├── models/                     # Implementation of the Grid-CNN and latent HMM modules
├── experiments/                # Training routines, configuration files, and evaluation scripts
├── notebooks/                  # Jupyter notebooks for exploratory analysis and visualization
├── results/                    # Experimental outputs, logs, and figures
├── paper/                      # LaTeX source files for the research paper
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or newer
- PyTorch (tested with version 1.10+)
- NumPy
- Other dependencies as listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hybrid-grid-cnn-hmm.git
   cd hybrid-grid-cnn-hmm
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate     # Linux/macOS
   venv\Scripts\activate        # Windows
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model

The training process is managed via scripts in the `experiments/` folder. A typical command to start training may look like:

```
python experiments/train.py --config experiments/configs/config_c4.yaml
```

The configuration file allows you to specify parameters such as candidate count (C=4 or C=8), learning rate, batch size, number of epochs, etc.

### Evaluating the Model

After training, you can evaluate the model on the test set by running:

```
python experiments/evaluate.py --model_path path/to/saved/model.pth
```

### Visualizations

For visualizing latent candidate activations, training loss curves, and other performance metrics, refer to the Jupyter notebooks in the `notebooks/` directory.

---

## Experiments

We performed experiments on the synthetic SPR benchmark dataset, which comprises sequences of abstract symbols annotated based on hidden poly‐rules (e.g., shape-count, color-position, parity, and order predicates). Two main configurations were evaluated:
- **C=4:** Achieved 56.00% Test Accuracy, with a Development Accuracy of 50.00%, Precision of 0.48, and Recall of 0.62.
- **C=8:** Achieved 54.00% Test Accuracy, with slight degradation in recall.

The experiments illustrate the trade-offs between latent candidate representation richness and model performance.

---

## Results

Key performance metrics:
- **Test Set Accuracy (C=4):** 56.00%
- **Development Set Accuracy:** 50.00%
- **Precision:** 0.48
- **Recall (C=4):** 0.62
- **Test Set Accuracy (C=8):** 54.00%
- **Recall (C=8):** 0.50

We provide detailed visualizations and analysis of candidate probability distributions and convergence behavior in the `results/` folder.

---

## Future Work

The current study lays the groundwork for a hybrid approach to SPR. Potential future directions include:
- Extended training regimes and hyperparameter tuning for enhanced performance.
- Exploring deeper and more complex CNN architectures.
- Implementing advanced regularization techniques (e.g., entropy-based penalties) for more distinct latent candidate activations.
- Integrating additional neuro-symbolic techniques for improved interpretability and performance.

---

## Contributing

Contributions to this repository are welcome! If you have ideas, bug fixes, or improvements, please feel free to submit an issue or create a pull request. For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

For additional details and the underlying theory, please refer to the research paper available in the `paper/` directory.

- Example references:
  - [Arxiv: Hybrid CNN-HMM for Lipreading](https://arxiv.org/abs/1906.12170v1)
  - [Arxiv: Writer-aware CNN-HMM for Handwritten Text Recognition](https://arxiv.org/abs/1812.09809v2)
  - [Arxiv: Neuro-symbolic Rule Extraction](https://arxiv.org/abs/2501.16677v1)

---

We hope you find this repository useful for exploring hybrid approaches in Symbolic Pattern Recognition. Thank you for your interest!