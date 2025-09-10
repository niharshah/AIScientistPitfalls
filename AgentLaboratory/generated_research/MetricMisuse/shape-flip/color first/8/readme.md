# Enhanced Neuro-Symbolic Representations in Machine Learning

This repository contains the code, experimental setup, and supporting materials for the research project “Exploring Enhanced Neuro-Symbolic Representations in Machine Learning.” The project investigates the integration of continuous deep learning feature extraction with explicit symbolic reasoning, targeting challenging visual symbolic reasoning tasks.

---

## Table of Contents

- [Overview](#overview)
- [Paper Summary](#paper-summary)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Experimental Details](#experimental-details)
- [Results and Analysis](#results-and-analysis)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project proposes a novel neuro-symbolic framework, named **Symbol-LLM**, which fuses continuous feature extraction (e.g., using logistic regression on aggregate features) with explicit symbolic rule-based reasoning. Our method formalizes a symbolic system (𝒮, ℛ) where each rule is expressed as a directed hyperedge:

  m₁ ∧ m₂ ∧ … ∧ mₙ ⊨ c

with logical entailment checked via a fuzzy logic scoring function τ(r) ≥ eₕ. The framework incorporates a “symbol-rule loop” that leverages large language models (LLMs) for automatic symbol extraction and rule extension.

---

## Paper Summary

The attached research paper provides a comprehensive description of the following aspects:

1. **Problem Motivation:**  
   - Limitations of standard statistical/deep models (System-1 approaches) for visual symbolic reasoning.
   - The need for interpretability and enhanced reasoning by integrating symbolic rules (System-2 processing).

2. **Methodology:**  
   - Formal definition of the symbolic system (𝒮, ℛ) and the entailment scoring mechanism.
   - Overview of the Symbol-LLM framework which iteratively extends symbolic rules using LLM prompts.
   - Integration of symbolic reasoning with aggregate continuous features (e.g., unique color/shape counts, token counts) using a baseline logistic regression model.

3. **Experimental Setup:**  
   - Detailed evaluation on the SPR_BENCH dataset.
   - Use of weighted metrics such as Standard Accuracy, Color-Weighted Accuracy (CWA), and Shape-Weighted Accuracy (SWA).
   - Analysis of results showing a baseline performance gap compared to state-of-the-art neuro-symbolic methods.

4. **Discussion and Future Research Directions:**  
   - Insights regarding feature representation limitations.
   - Prospective improvements using richer, sequential, and relational feature extraction integrated with neuro-symbolic components.

For further technical details, please refer to the [research paper](./paper.pdf) included in this repository.

---

## Repository Structure

    .
    ├── data/                   # SPR_BENCH dataset and associated files
    ├── experiments/            # Scripts for running experiments and evaluating models
    ├── notebooks/              # Jupyter notebooks for exploratory data analysis and ablation studies
    ├── src/                    # Source code for the neuro-symbolic model and logistic regression baseline
    │   ├── model.py            # Implementation of the Symbol-LLM framework & baseline model
    │   ├── symbolic_system.py  # Formalization and utilities for symbolic rule manipulation
    │   └── feature_extraction.py # Functions for extracting aggregate features from data
    ├── paper.pdf               # Full research paper PDF
    ├── README.md               # This readme file
    └── requirements.txt        # Python package dependencies

---

## Setup and Installation

To set up the repository locally, please follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/neuro-symbolic-representations.git
   cd neuro-symbolic-representations
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows, run `venv\Scripts\activate`
   ```

3. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

   The dependencies include standard libraries such as NumPy, scikit-learn, PyTorch (if applicable for LLM interfacing), and others necessary for processing and analysis.

---

## Usage

### Running the Baseline Experiment

The baseline logistic regression model uses aggregate features (number of unique colors, shapes, and token count) from the SPR_BENCH dataset. To run the baseline experiment, execute:

```bash
python experiments/run_baseline.py --data_dir ./data/spr_bench --output_dir ./results
```

### Running the Neuro-Symbolic Module

The Symbol-LLM framework that implements the symbolic rule extension and fuzzy entailment verification can be run as follows:

```bash
python experiments/run_symbol_llm.py --data_dir ./data/spr_bench --output_dir ./results
```

These scripts will produce logs, evaluation metrics, and visualizations (such as confusion matrices) in the designated output directory.

### Jupyter Notebooks

For exploratory analysis and visualization of results, see the notebooks in the `notebooks/` folder. For example:

- `notebooks/EDA.ipynb`: Exploratory Data Analysis on symbolic sequences.
- `notebooks/ablation_studies.ipynb`: Examining the impact of different aggregate features.

---

## Experimental Details

- **Dataset:** SPR_BENCH – a visual symbolic reasoning dataset with tokens representing shapes and colors.
- **Feature Extraction:** Computation of unique color counts, unique shape counts, and total token counts per instance.
- **Models:**  
  - **Baseline:** Logistic regression using aggregate features.
  - **Neuro-Symbolic:** Integration of LLM-based symbolic rule extraction with logistic regression.
- **Metrics:**  
  - Standard Accuracy  
  - Color-Weighted Accuracy (CWA)  
  - Shape-Weighted Accuracy (SWA)

For a complete description of the methodology and experimental protocol, please refer to the [research paper](./paper.pdf).

---

## Results and Analysis

The baseline logistic regression model reported:
- **Test Accuracy:** 54.41%
- **Color-Weighted Accuracy (CWA):** 54.53%
- **Shape-Weighted Accuracy (SWA):** 52.87%

Detailed error analysis and visualizations (e.g., confusion matrices) are generated during the experiments and stored in the output directory for further examination. The observed performance gap compared to advanced neuro-symbolic methods (65.0% CWA and 70.0% SWA) motivates the integration of richer feature representations and enhanced symbolic reasoning techniques.

---

## Future Work

The future research agenda includes:
- **Enhanced Neuro-Symbolic Integration:** Deeper integration of LLM-driven symbolic rule extraction with continuous representations using architectures like transformers.
- **Relational Feature Extraction:** Incorporation of higher-order statistics, n-gram analyses, and graph-based connectivity to improve capturing relational dependencies.
- **Robust Regularization Techniques:** Improved methodologies to mitigate overfitting and enhance generalization.
- **Extensive Empirical Evaluation:** Broader cross-validation studies and statistical tests (e.g., McNemar's test) to validate performance improvements.
- **Scalability and Efficiency:** Optimization of the computational pipeline to seamlessly handle larger datasets and complex symbolic reasoning.

---

## Acknowledgments

We would like to thank the research community for insightful discussions on neuro-symbolic integration. Special thanks to the anonymous reviewers and collaborators who provided invaluable feedback during the development of this project.

---

For more details, please refer to the full [research paper](./paper.pdf). Contributions and feedback are welcome; please open issues or submit pull requests to help improve this project.

Happy coding!