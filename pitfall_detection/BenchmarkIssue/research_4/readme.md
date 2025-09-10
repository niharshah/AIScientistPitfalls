# Advances in Symbolic Pattern Recognition

Welcome to the repository for the "Advances in Symbolic Pattern Recognition" project. This project presents a robust baseline for the task of symbolic pattern recognition (SPR) by leveraging a simple logistic regression model combined with TF-IDF feature extraction on character-level n-grams. The goal is to decide whether a given symbolic sequence conforms to an underlying hidden rule. The repository includes code, experimental setups, analysis reports, and associated figures.

---

## Table of Contents

- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Methodology](#methodology)
- [Datasets and Experimental Setup](#datasets-and-experimental-setup)
- [Results](#results)
- [Discussion and Future Work](#discussion-and-future-work)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)
- [Citations](#citations)

---

## Overview

This repository documents our research on employing a baseline statistical method for symbolic pattern recognition (SPR). Our approach uses a logistic regression model defined as:

  ŷ = sgn(wᵀ x + b)

and is trained using a mean squared error loss function:

  L = (1/n) Σᵢ (yᵢ − sgn(wᵀ xᵢ + b))²

By extracting TF-IDF features from character-level n-grams (n = 2, 3, 4), the model achieves competitive performance across multiple benchmark datasets. Our experimental results are summarized as follows:

- **EWERV:** Test Accuracy = 80.5%
- **URCJF:** Test Accuracy = 84.7%
- **PHRTV:** Test Accuracy = 94.5%
- **IJSJF:** Test Accuracy = 74.0%

The repository contains the paper write-up, related code, and additional figures including accuracy bar charts and confusion matrices, which provide insight into misclassifications (notably for minority classes).

---

## Project Motivation

Symbolic pattern recognition is central to tasks that require detecting and reasoning about abstract rules embedded within symbolic sequences. Although advanced models have been developed, our work validates that even simple, interpretable methods—when appropriately designed—can deliver robust performance. This project serves as a foundation for further exploration into integrating dynamic symbol binding, external-memory architectures, and enriched feature representations.

---

## Methodology

1. **Feature Extraction:**  
   - Raw symbolic sequences are converted into numerical feature vectors via TF-IDF extraction on character-level n-grams.
   - The selected n-gram ranges are 2, 3, and 4, with a configurable vocabulary size (set to 500 in our experiments).

2. **Model Definition:**  
   - We use a logistic regression model with decision function:  
  ŷ = sgn(wᵀ x + b)
   - The model is trained using a mean squared error (MSE) loss function.

3. **Training and Evaluation:**  
   - The datasets are split into 2000 training, 500 development, and 1000 test samples.
   - Performance is quantified using accuracy and confusion matrix analyses to capture class-specific error trends (e.g., for minority classes).

4. **Implementation:**  
   - The code is implemented in Python, using libraries such as scikit-learn for logistic regression and TF-IDF computations.
   - Additional packages include HuggingFace datasets for data loading and matplotlib for visualization of results.

---

## Datasets and Experimental Setup

- **Datasets:**  
  Four benchmark datasets (EWERV, URCJF, PHRTV, and IJSJF) are included, with each dataset containing:
  - 2000 training samples
  - 500 development samples
  - 1000 test samples

- **Experimental Details:**  
  - n-gram range: (2, 4)
  - Maximum feature count: 500
  - Maximum iterations for logistic regression solver: 1000  
  - Evaluation metrics include accuracy (percentage of correctly classified samples) and confusion matrices, particularly examining the PHRTV dataset for class-specific insights.

---

## Results

Our baseline model achieves the following accuracies:

| Dataset | Train Acc (%) | Dev Acc (%) | Test Acc (%) |
|---------|---------------|-------------|--------------|
| EWERV   | 82.10         | 80.20       | 80.50        |
| URCJF   | 86.40         | 84.80       | 84.70        |
| PHRTV   | 94.80         | 95.60       | 94.50        |
| IJSJF   | 79.65         | 72.80       | 74.00        |

Ablation studies indicate that removing TF-IDF feature extraction or altering the n-gram configurations results in noticeable performance degradation, especially for datasets with more subtle symbolic variations.

**Visualizations:**  
- **Figure 1:** Bar chart summarizing test accuracies across benchmarks.  
- **Figure 2:** Confusion matrix for the PHRTV dataset highlighting that misclassifications predominantly occur in minority classes.

---

## Discussion and Future Work

### Discussion

- **Strengths:**  
  - The simplicity and transparency of the model facilitate clear diagnosis and interpretation.
  - Competitive performance is achieved on multiple benchmarks with minimal overfitting.

- **Limitations:**  
  - The TF-IDF approach may fail to capture higher-order dependencies or subtle symbolic variations.
  - The deterministic logistic regression model might not be well-suited for complex, non-linear patterns.

### Future Work

- **Enhanced Feature Representations:**  
  - Expanding feature representations beyond basic n-grams (e.g., skip-grams, variable-length tokenization).

- **Dynamic Symbol Binding:**  
  - Integrating external-memory modules or attention-based mechanisms to capture long-range dependencies.

- **Hybrid Models:**  
  - Exploring a combination of deterministic models with probabilistic or neural-symbolic components.
  - Developing meta-learning strategies to enable one-shot learning for evolving symbolic rules.

- **Evaluation Metrics:**  
  - Incorporating more nuanced metrics that consider the severity of misclassifications and decision boundary interpretability.

---

## Usage

To get started with the repository:

1. **Clone the Repository:**

   ```
   git clone https://github.com/yourusername/Advances-in-Symbolic-Pattern-Recognition.git
   cd Advances-in-Symbolic-Pattern-Recognition
   ```

2. **Setup the Environment:**

   Create a virtual environment and install the required dependencies:

   ```
   python -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Experiments:**

   Execute the main script to train and evaluate the model:

   ```
   python main.py
   ```

   This script runs the training pipeline, performs TF-IDF feature extraction on your datasets, trains the logistic regression model, and generates evaluation metrics and visualizations.

---

## Dependencies

The project requires the following key Python libraries:

- Python 3.x
- scikit-learn
- numpy
- pandas
- matplotlib
- HuggingFace datasets (for data loading)
- seaborn (optional, for enhanced visualizations)

A complete list of dependencies is available in the `requirements.txt` file.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citations

If you use or extend this work, please cite the following relevant publications and reports:

- [arXiv:1710.00077v1](https://arxiv.org/abs/1710.00077)
- [arXiv:2004.13577v1](https://arxiv.org/abs/2004.13577)
- [arXiv:2505.06745v1](https://arxiv.org/abs/2505.06745)
- Additional references mentioned within the accompanying research report.

---

Thank you for your interest in our work on symbolic pattern recognition. For any questions or further information, please open an issue or contact the project maintainers.

Happy coding!