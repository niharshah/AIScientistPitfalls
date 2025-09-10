# SPR-Based Sequence Recognition

This repository contains the code, documentation, and experiments for the research work titled **"A Comprehensive Analysis of SPR-Based Sequence Recognition"**. The work establishes a baseline approach for Symbolic Pattern Recognition (SPR) tasks using a count-based feature extraction strategy in combination with logistic regression. The study provides a clear formulation of SPR using simple yet interpretable features, detailed experimental evaluations, and a discussion on limitations and future enhancements.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Future Directions](#future-directions)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

In this work, the SPR task is approached as a binary classification problem on symbolic sequences. Each sequence comprises tokens defined by their shape and color. Our method computes a three-dimensional feature vector consisting of:
- **Number of unique shapes**
- **Number of unique colors**
- **Total token count**

This feature vector is then fed into a logistic regression model:
  
  ŷ = σ(Wx + b)

Despite the simplicity of the method, our experiments on the SPR_BENCH dataset indicate that while this linear model yields moderate performance (with standard accuracies of 59.14% on the development set and 54.25% on the test set), it also highlights the inherent challenges of capturing non-linear interactions in symbolic data. The repository includes additional experiments with augmented feature spaces (introducing interaction and quadratic terms) and detailed error analyses.

---

## Motivation

Symbolic Pattern Recognition (SPR) bridges classical symbolic methods and modern machine learning techniques. The primary goals of our research include:

- Providing an interpretable baseline model to understand the challenges in SPR.
- Highlighting the limitations of count-based features when capturing the latent structure in sequences.
- Setting a benchmark for future architectures, such as advanced non-linear models and neural-symbolic integration, to further improve performance metrics (e.g., target Closed-loop Weighted Accuracy of 65.0%).

---

## Repository Structure

The repository is structured as follows:

```
SPR-Sequence-Recognition/
├── data/
│   ├── SPR_BENCH/             # Dataset used in experiments (train, dev, test splits)
├── docs/
│   └── paper.pdf              # Research report (compiled from LaTeX source)
├── experiments/
│   ├── baseline.ipynb         # Jupyter Notebook for baseline logistic regression experiments
│   ├── ablation_study.ipynb   # Notebook for augmented feature experiments and error analysis
├── src/
│   ├── feature_extraction.py  # Code for extracting count-based features from input sequences
│   ├── model.py               # Implementation of logistic regression and feature augmentation
│   └── utils.py               # Utility functions (data loading, evaluation metrics, plotting)
├── results/
│   ├── figures/               # Diagnostic plots (histograms, confusion matrices)
│   └── performance_metrics.csv# Summary of the experimental results
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── LICENSE                    # Licensing information
```

---

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/SPR-Sequence-Recognition.git
   cd SPR-Sequence-Recognition
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The key dependencies include:
   - Python 3.7+
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - Jupyter Notebook (for running experiments)

---

## Usage

### Data Preparation

Place the SPR_BENCH dataset (comprising training, development, and test splits) in the `data/SPR_BENCH/` directory. Ensure the sequences are formatted with tokens that represent shape and color.

### Running the Baseline Experiment

To run the baseline logistic regression experiments, open the Jupyter Notebook located in `experiments/baseline.ipynb` and execute the cells sequentially. The notebook contains:

- Data loading and preprocessing routines.
- Feature extraction using count-based methods.
- Model training using scikit-learn's LogisticRegression.
- Evaluation metrics: Standard Accuracy and Shape-Weighted Accuracy (SWA).
- Diagnostic plots (histograms and confusion matrices).

### Extended Experiments

For experiments with augmented features (interaction and quadratic terms), refer to the `experiments/ablation_study.ipynb` notebook. This notebook explores:

- Enhanced feature mappings φ(x) ∈ ℝ⁹.
- Comparative analysis with the baseline model.
- Discussion on performance improvements and limitations.

---

## Experiments & Results

Our experimental findings include:

- **Development Set:**
  - Standard Accuracy: 59.14%
  - Shape-Weighted Accuracy (SWA): 0.5857
- **Test Set:**
  - Standard Accuracy: 54.25%
  - SWA: 0.5411

Diagnostic analyses reveal that sequences featuring intermediate shape diversity are particularly prone to misclassification. Despite minor gains from augmented features, the baseline highlights the need for more sophisticated methods (e.g., kernel methods or hybrid neural-symbolic architectures) to fully capture non-linear interactions.

Detailed results and plots can be found in the `results/` folder.

---

## Future Directions

The research outlines several promising directions for further development:

- **Enhanced Feature Engineering:** Incorporate positional encodings and spatial relationships in the token sequences.
- **Non-Linear Modeling:** Explore advanced non-linear models like kernel-based methods or tree ensembles.
- **Hybrid Neural-Symbolic Integration:** Combine the expressivity of deep learning with the interpretability of symbolic reasoning.
- **Robust Evaluation:** Implement extensive statistical testing and cross-validation protocols to better assess performance.

---

## Acknowledgments

We sincerely thank the research community for their continued contributions to symbolic pattern recognition and machine learning. This work builds on a rich legacy of previous studies, and we hope the insights and benchmarks provided here will serve as a strong foundation for future advancements in SPR.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, open issues, or contact the maintainers for further discussion on this research. Happy exploring!