# Symbolic Pattern Recognition Baseline

This repository contains the code, experimental setup, and supplementary materials for our research on symbolic pattern recognition. The work centers on a baseline model that leverages a multilayer perceptron (MLP) trained on TF–IDF features, optimized with a composite loss function designed to promote interpretable, sparse, and nearly binary latent representations. The repository also includes the full LaTeX research paper detailing our methodology, experiments, and analysis on the SPR_BENCH dataset.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results and Discussion](#results-and-discussion)
- [Repository Structure](#repository-structure)
- [Installation and Usage](#installation-and-usage)
- [Citation](#citation)
- [License](#license)

---

## Overview

Symbolic pattern recognition is at the intersection of machine learning and formal logic, with significant applications in areas such as natural language processing, computer vision, and automated reasoning. This repository provides:

- A baseline approach using an MLP with TF–IDF features.
- A composite loss function integrating supervised contrastive loss, entropy minimization, and L1 sparsity regularization.
- Detailed experiments on the SPR_BENCH dataset with evaluation metrics such as overall accuracy and shape-weighted accuracy (SWA).
- Comprehensive documentation, error analysis (including histogram and confusion matrix visualizations), and discussion on limitations and future directions.

The goal is to create a transparent and interpretable framework that can serve as a stepping stone towards more advanced hybrid neuro-symbolic methods.

---

## Background

For applications requiring explainability, traditional deep-learning models often fall short in interpretability. This work builds upon prior research in neuro-symbolic integration by:

- Representing inputs as TF–IDF vectors derived from symbolic sequences—each token corresponds to a geometric shape.
- Encouraging sparse and near-binary representations that allow for straightforward rule extraction.
- Using supervised contrastive loss to cluster similar inputs in the latent space, thus reinforcing the symbolic structures.

---

## Methodology

The main components of the methodology are:

1. **Data Representation:**  
   - Inputs are transformed using TF–IDF representations that capture both word frequency and symbolic nuances.
   - Care is taken to ensure that tokens, including single-character tokens representing geometric shapes, are preserved.

2. **Model Architecture:**  
   - A multilayer perceptron (MLP) with one hidden layer of 100 neurons using ReLU activation.
   - An output layer with sigmoid activation to ensure neuron activations lie between 0 and 1, facilitating binarization.

3. **Composite Loss Function:**  
   The total loss is defined as:
   L_Total = α * L_supcon + β * L_entropy + γ * L_sparsity,  
   where:
   - L_supcon: Ensures clustering of similar inputs.
   - L_entropy: Drives activations toward binary values by entropy minimization.
   - L_sparsity: Promotes a sparse latent representation via L1 regularization.

4. **Post-Training Binarization:**  
   - The continuous latent activations are thresholded to yield binary vectors, which serve as proxies for symbolic predicates.

---

## Experimental Setup

Experiments were performed on the SPR_BENCH dataset:

- **Dataset:**  
  - Training samples: 20,000  
  - Development samples: 5,000  
  - Test samples: 10,000  
  - Each sample is a CSV record comprising a unique identifier, a symbolic sequence, and a label.

- **Training Details:**  
  - Optimizer: Adam  
  - Maximum iterations: 300  
  - Fixed random seed for reproducibility  
  - Hyperparameters (α, β, and γ) tuned via cross-validation.

- **Evaluation Metrics:**  
  1. Overall Accuracy: Percentage of correctly predicted samples.
  2. Shape-Weighted Accuracy (SWA): Weights correctness by the number of unique shape tokens per sequence.  
     Formula:  
     SWA = (Σ_i w_i * I(y_i = ŷ_i)) / (Σ_i w_i)

- **Visualizations:**  
  - Histogram Analysis of unique shape counts vs. prediction accuracy.
  - Confusion Matrix detailing misclassification patterns.

A summary of experimental parameters is provided in the paper, along with a detailed ablation study demonstrating the contribution of each loss component.

---

## Results and Discussion

- **Performance:**  
  - Development Set Accuracy: 55.12%  
  - Test Set Accuracy: 56.88%  
  - Development Set SWA: 0.54  
  - Test Set SWA: 0.55

- **Key Findings:**  
  - The baseline model underperforms relative to state-of-the-art targets (~70% accuracy, SWA of 0.65).
  - Error analysis indicates performance degradation with increased symbolic complexity (i.e., higher unique shape counts).
  - Ablation studies demonstrate the importance of both entropy minimization and supervised contrastive loss for enhancing performance and interpretability.

- **Future Directions:**  
  - Integrating rule-extraction layers for direct mapping to symbolic predicates.
  - Incorporating enhanced non-linear modeling (e.g., attention mechanisms) while maintaining interpretability.
  - Exploring self-supervised pretraining and hybrid neuro-symbolic approaches to bridge the performance gap.

---

## Repository Structure

The repository is organized as follows:

├── data/  
│   └── SPR_BENCH/         # SPR_BENCH dataset (CSV files)  
├── experiments/  
│   ├── error_analysis/     # Histogram and confusion matrix visualizations  
│   └── ablation_study/     # Scripts for loss function ablation studies  
├── models/  
│   └── mlp_model.py        # MLP model implementation and training scripts  
├── paper/  
│   └── symbolic_pattern_recognition.tex  # LaTeX source of the research paper  
├── README.md  
└── requirements.txt        # Python dependencies  

---

## Installation and Usage

### Prerequisites

- Python 3.7+
- Recommended: Anaconda or virtualenv for dependency management

### Installation

1. Clone the repository:

   git clone https://github.com/your-username/symbolic-pattern-recognition.git  
   cd symbolic-pattern-recognition

2. Install the required packages:

   pip install -r requirements.txt

   The requirements include libraries for numerical computation (NumPy, SciPy), machine learning (scikit-learn, PyTorch/TensorFlow depending on implementation), and plotting (Matplotlib, Seaborn).

### Running the Model

1. Prepare the dataset:  
   Place the SPR_BENCH CSV files in the `data/SPR_BENCH/` directory.

2. Train the model:
   
   python models/mlp_model.py --train

   Additional command-line options (e.g., iterations, hyperparameters) can be found by running:

   python models/mlp_model.py --help

3. Evaluate the model:

   python models/mlp_model.py --evaluate

4. Generate visualizations:  
   Navigate to the `experiments/error_analysis/` folder and run the provided scripts to reproduce the histogram and confusion matrix plots.

---

## Citation

If you find this work useful for your research, please consider citing our paper:

Agent Laboratory. (2023). Research Report: A Preliminary Analysis of Symbolic Pattern Recognition. Retrieved from [repository URL].

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions, issues, or contributions, please feel free to open an issue or submit a pull request. Happy coding!