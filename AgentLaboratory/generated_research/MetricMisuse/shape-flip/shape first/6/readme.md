# Unveiling Chained Transformations in Model-Driven Engineering

This repository contains the code, experiments, and documentation associated with the research report titled **"Unveiling Chained Transformations in Model-Driven Engineering"**. The work focuses on symbolic pattern recognition (SPR) using the SPR_BENCH dataset and investigates the limitations of a baseline logistic regression model that leverages surface-level features to capture latent symbolic dependencies.

---

## Overview

In this project, we explore the challenge of modeling chaining constraints in SPR tasks. We provide:

- A baseline logistic regression model implemented in Python.
- A detailed experimental setup using the SPR_BENCH dataset.
- Code for feature extraction that computes:
  - **Shape Complexity:** Number of unique shape tokens.
  - **Color Complexity:** Number of unique color tokens.
  - **Token Count:** Total number of tokens in a sequence.
- Evaluation metrics including standard accuracy and a novel **Shape-Weighted Accuracy (SWA)** which assigns greater importance to sequences with a richer symbolic vocabulary.
- A comprehensive report discussing limitations, error analysis, and promising future directions such as candidate rule extraction via large language models (LLMs), inductive logic programming (ILP), and ensemble techniques like Elite Bases Regression (EBR).

---

## Repository Structure

- **/data/**  
  Contains instructions or sample scripts for loading the SPR_BENCH dataset. (Note: Due to licensing or size, the dataset might need to be downloaded separately.)

- **/src/**
  - `feature_extraction.py`: Implements tokenization and computes surface-level features (shape complexity, color complexity, token count).
  - `model.py`: Contains the logistic regression baseline model.
  - `train.py`: Script for training and evaluating the model with options to compute both standard accuracy and SWA.
  - `utils.py`: Helper functions for computing evaluation metrics and generating visualizations (e.g., histograms, confusion matrix).

- **/docs/**
  - The research report (written in LaTeX) detailing methodology, results, and discussion.

- **README.md**  
  This file provides an overview of the project.

---

## Dataset: SPR_BENCH

The SPR_BENCH dataset is designed to challenge models with latent symbolic dependencies and chaining constraints. Key details include:
- **Training set:** 20,000 examples
- **Development set:** 5,000 examples
- **Test set:** 10,000 examples

Each example in the dataset contains:
- A unique identifier
- A token sequence (from which shape/color features are derived)
- A corresponding binary label

---

## Methodology

### Feature Extraction
The feature extraction process involves:
- **Tokenization:** Splitting the input sequence by whitespace.
- **Shape and Color Extraction:**  
  - The first character of each token is treated as the shape indicator.
  - Remaining characters are used for deducing color complexity.
- **Computations:**  
  - **Shape Complexity:** Count of unique shape tokens.
  - **Color Complexity:** Count of unique color tokens.
  - **Token Count:** Total number of tokens.

### Baseline Model
A logistic regression model is trained using the three-dimensional feature vector:
  
  f(x) = σ(wᵀx + b)

where:
- x ∈ ℝ³ (the feature vector),
- σ is the sigmoid function, and
- the decision rule is:  
  ŷ = 1 {f(x) > 0.5}

The model is trained using cross-entropy loss optimized via gradient descent (using scikit-learn’s implementation).

### Evaluation Metrics
- **Standard Accuracy**
  
  Accuracy = (1/N) Σ 1{yᵢ = ŷᵢ}

- **Shape-Weighted Accuracy (SWA)**
  
  SWA = (Σ wᵢ · 1{yᵢ = ŷᵢ}) / (Σ wᵢ)  
  Here, wᵢ is the number of unique shape tokens for the i-th sample.

---

## Experimental Results

The baseline model performance on the SPR_BENCH dataset is summarized as follows:

| Metric                 | Development Set (%) | Test Set (%) |
| ---------------------- | ------------------- | ------------ |
| **Standard Accuracy**  | 54.84               | 56.52        |
| **Shape-Weighted Accuracy** | 53.57         | 55.32        |

### Observations:
- The baseline model, while stable, underperforms compared to state-of-the-art (SOTA) benchmarks (70.00% standard accuracy and 65.00% SWA).
- Ablation studies suggest that removing any of the features (shape complexity, color complexity, or token count) causes an additional ~3% decline in performance.
- Confusion matrix analysis indicates that the model struggles particularly with sequences exhibiting low variance in shape complexity.

---

## How to Run

### Prerequisites

- Python 3.7+
- [scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- Additional libraries (e.g., matplotlib for visualizations)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/SPR_ChainedTransformations.git
   cd SPR_ChainedTransformations
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Experiments

1. **Feature Extraction & Model Training:**
   ```
   python src/train.py --data_path path/to/SPR_BENCH_dataset
   ```

2. **Evaluation & Visualization:**
   The training script will automatically compute the standard accuracy and SWA, and generate visualizations (e.g., histograms, confusion matrix) saved in the `/results/` folder.

---

## Future Work

The current study serves as a benchmark for SPR tasks using surface-level features. Future research directions include:

- **Advanced Feature Extraction:**  
  Incorporate context-aware embeddings and graph-based representations to capture higher-order dependencies.

- **Hybrid Models:**  
  Integrate large language model (LLM) generated candidate rules with iterative refinement via inductive logic programming (ILP).

- **Ensemble Strategies:**  
  Explore methods like Elite Bases Regression (EBR) to combine the strengths of local approximators with deep neural components.

- **Comprehensive Evaluation:**  
  Perform cross-validation, hyperparameter tuning, and testing on diverse SPR datasets to ensure robust and generalizable performance.

---

## Citation

If you use this work in your research, please cite:

    @misc{agentlab2023,
      author = {Agent Laboratory},
      title = {Unveiling Chained Transformations in Model-Driven Engineering},
      year = {2023},
      note = {Preprint available at arXiv or your publication link}
    }

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions or collaborations, please reach out to:
- [Your Name or Research Group]
- Email: your.email@example.com

Happy exploring and thank you for your interest!
