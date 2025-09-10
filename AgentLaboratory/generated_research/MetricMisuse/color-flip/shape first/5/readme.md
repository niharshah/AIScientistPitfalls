# SPR_BENCH Baseline Analysis

This repository contains the code, experiments, and research report associated with the baseline analysis for symbolic pattern recognition on the SPR_BENCH dataset using a bag-of-tokens approach. Our study employs a simple Logistic Regression classifier applied on CountVectorizer-based token frequency features and introduces a novel evaluation metric called Shape-Weighted Accuracy (SWA) to better capture the variability present in symbolic sequences.

---

## Overview

Symbolic pattern recognition is fundamental in areas such as natural language processing, bioinformatics, robotics, and automated reasoning. Datasets like SPR_BENCH challenge standard approaches by exhibiting complex sequential and structural dependencies. In this work, we establish a baseline using:

- **Bag-of-Tokens Representation:** Each symbolic sequence is converted into a high-dimensional sparse vector where each dimension corresponds to token frequency.
- **Logistic Regression Classifier:** A straightforward linear model that provides interpretability and efficiency.
- **Shape-Weighted Accuracy (SWA):** A performance metric that weighs each sequence by its number of unique shape types, ensuring that sequences with greater symbolic diversity have a proportionately larger impact on the overall accuracy measure.

The goal of this baseline is to provide a transparent, reproducible starting point for future research while also highlighting the limitations of flat token counts when compared to more sophisticated, structure-aware models.

---

## Repository Structure

Here is an overview of the repository structure:

```
SPR_BENCH-Baseline-Analysis/
├── data/
│   └── [SPR_BENCH_dataset.csv]           # CSV dataset file(s)
├── notebooks/
│   └── SPR_BENCH_baseline_analysis.ipynb   # Jupyter notebook for exploratory analysis and visualization
├── src/
│   ├── data_loader.py                      # Code to load and preprocess the dataset using HuggingFace
│   ├── feature_extraction.py               # Implementation of the bag-of-tokens approach using CountVectorizer
│   ├── logistic_regression.py              # Logistic Regression model training and evaluation code
│   └── metrics.py                          # Calculation of standard accuracy and Shape-Weighted Accuracy (SWA)
├── figures/
│   ├── confusion_matrix.png                # Confusion matrix visualization
│   └── roc_curve.png                       # ROC curve and AUC plot
├── paper/
│   └── SPR_BENCH_Baseline_Analysis.pdf       # PDF version of the full research report
├── README.md                               # This file
└── requirements.txt                        # Python dependencies and libraries
```

---

## Installation

To set up the project locally, please follow these steps:

1. **Clone the repository:**

   ```
   git clone https://github.com/<username>/SPR_BENCH-Baseline-Analysis.git
   cd SPR_BENCH-Baseline-Analysis
   ```

2. **Create a virtual environment (optional but recommended):**

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - Python 3.7+
   - scikit-learn
   - HuggingFace datasets
   - NumPy, pandas
   - matplotlib (for visualization)
   - Jupyter Notebook (if using the provided notebook)

---

## Usage

### Running the Analysis

The code in the `src/` directory implements the complete analysis pipeline:

1. **Data Loading & Preprocessing:**

   The file `data_loader.py` loads the SPR_BENCH dataset (provided in CSV format) using the HuggingFace datasets library and prepares the data splits (training, development, and test sets).

2. **Feature Extraction:**

   - `feature_extraction.py` utilizes scikit-learn’s `CountVectorizer` to tokenize the sequences and produces high-dimensional bag-of-tokens representations.

3. **Model Training & Evaluation:**

   - `logistic_regression.py` implements the training of the Logistic Regression classifier.
   - The classifier is optimized by minimizing the cross-entropy loss.
   - Performance metrics are computed on both the development and test sets.

4. **Metrics:**

   - `metrics.py` contains the functions to compute:
     - **Standard Accuracy:** The proportion of correctly classified sequences.
     - **Shape-Weighted Accuracy (SWA):** An adjusted accuracy metric that weights each sequence by its number of unique shape types.

### Running the Pipeline

You can run the entire pipeline with:

```
python src/logistic_regression.py
```

Alternatively, if you prefer to work interactively, open the Jupyter notebook in the `notebooks/` directory:

```
jupyter notebook notebooks/SPR_BENCH_baseline_analysis.ipynb
```

This notebook provides an interactive walkthrough of:
- Data loading
- Feature extraction
- Model training
- Visualizations (confusion matrix, ROC curve)
- Performance evaluation on both accuracy metrics.

---

## Experimental Results

On the SPR_BENCH dataset, the baseline yielded the following results:

- **Development Set:**
  - Standard Accuracy: 58.40%
  - SWA: 58.82%
- **Test Set:**
  - Standard Accuracy: 59.89%
  - SWA: 60.40%

These findings demonstrate that while a simple bag-of-tokens approach can capture relevant token frequency information, there remains room for improved performance when incorporating sequence-dependent and structural features.

Visualizations such as the confusion matrix and ROC curve (provided in the `figures/` directory) help illustrate the misclassification patterns and discriminative capacity of the classifier.

---

## Future Work

While this baseline provides a solid starting point, several avenues for future research are promising:

- **Incorporating Sequential Dependencies:** Integrating models (e.g., recurrent neural networks or transformers) to capture the order and contextual relationships among tokens.
- **Augmenting Feature Representations:** Combining bag-of-tokens features with additional context-aware features driven by symbolic rules or logical representations (e.g., Logical Hidden Markov Models).
- **Enhanced Evaluation Metrics:** Extending the Shape-Weighted Accuracy metric and exploring other domain-specific performance measures.
- **Explainability:** Leveraging explainable AI techniques to better understand misclassification patterns and improve model interpretability.

---

## Citation

If you find this work useful in your research, please consider citing our work:

[SPR_BENCH Baseline Analysis: Baseline Analysis for SPR_BENCH Dataset Using a Bag-of-Tokens Approach.](./paper/SPR_BENCH_Baseline_Analysis.pdf)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or contributions, please reach out by opening an issue on GitHub or contacting the repository maintainer at [email@example.com](mailto:email@example.com).

Happy coding and research!
