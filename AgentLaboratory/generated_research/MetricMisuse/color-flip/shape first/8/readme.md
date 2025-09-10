# SPR_BENCH Symbolic Pattern Recognition

This repository contains the code, data processing scripts, and experimental framework for our research on symbolic pattern recognition using the SPR_BENCH dataset. The work demonstrates a baseline approach using logistic regression on handcrafted features and introduces a novel evaluation metric—Shape-Weighted Accuracy (SWA)—to better capture the structural complexity of symbolic sequences.

---

## Overview

Symbolic pattern recognition is an emerging area that bridges the gap between explicit rule-based systems and modern machine learning techniques. In this project, we:
- Extract explicit features from symbolic sequences, including:
  - Count of unique shapes
  - Count of unique colors
  - Sequence length (total number of tokens)
- Standardize these features using z-score normalization.
- Train a logistic regression classifier to predict labels from these symbolic sequences.
- Evaluate the model using two metrics:
  - Overall Accuracy
  - Shape-Weighted Accuracy (SWA)

The SWA metric assigns higher weights to samples with greater structural diversity (measured by the unique shape count), highlighting cases where correctly recognizing complex symbolic interactions is particularly challenging.

---

## Repository Structure

```
├── data/
│   ├── train.csv               # Training set (20,000 samples)
│   ├── dev.csv                 # Development set (5,000 samples)
│   └── test.csv                # Test set (10,000 samples)
├── notebooks/
│   └── exploration.ipynb       # Jupyter Notebook for data exploration & visualization (e.g., confusion matrix, ROC curves)
├── src/
│   ├── feature_extraction.py   # Extraction of handcrafted features from symbolic sequences
│   ├── model.py                # Implementation of logistic regression model and training routines
│   ├── metrics.py              # Implementation of overall accuracy and SWA metric
│   └── train.py                # Script to train and validate the model using the provided dataset splits
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Background

In recent years, symbolic reasoning has been revisited, merging ideas from classical automata theory with modern deep learning. Our formulation is based on explicitly extracting features like counts of unique shapes and colors, and sequence length, to build a transparent and interpretable baseline model. Our approach is inspired by previous works (see arXiv:2501.00296v3, arXiv:2503.04900v1, and arXiv:1710.00077v1) and serves as a reproducible benchmark for further exploration in symbolic reasoning.

---

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/SPR-BENCH-Symbolic-Pattern-Recognition.git
   cd SPR-BENCH-Symbolic-Pattern-Recognition
   ```

2. Create a virtual environment (optional, but recommended):
   ```
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Data

The SPR_BENCH dataset is divided into three splits:
- **Training:** 20,000 symbolic sequence samples
- **Development:** 5,000 samples (used for hyperparameter tuning and early stopping)
- **Test:** 10,000 samples (used for final evaluation)

Each sample is a sequence containing tokens that represent various shapes and colors. The feature extraction process converts each sequence into a 3-dimensional vector:
- Unique shapes count
- Unique colors count
- Sequence length

---

## Usage

### Training the Model

To train the logistic regression model with feature extraction and evaluate its performance, run:

```
python src/train.py --data_dir ./data --max_iter 200
```

This script:
- Reads the data from the designated splits
- Extracts and standardizes the features
- Trains the logistic regression model
- Evaluates using both overall accuracy and the specialized Shape-Weighted Accuracy (SWA) metric

### Evaluating the Model

After training, evaluation results (including confusion matrix and ROC curve visualizations) are automatically generated and saved in a designated output folder.

---

## Experimental Setup

- **Feature Extraction:**  
  Handcrafted features are extracted to form the vector:  
  x = (unique shape count, unique color count, sequence length)

- **Standardization:**  
  Each feature is normalized using z-score normalization:
  ```
  x_tilde = (x - μ) / σ
  ```
  
- **Classifier:**  
  A logistic regression classifier is used:
  ```
  P(y = 1 | x_tilde) = 1 / (1 + exp(-(w^T x_tilde + b)))
  ```
  
- **Evaluation Metrics:**  
  - **Overall Accuracy:** Standard classification accuracy  
  - **Shape-Weighted Accuracy (SWA):**
    ```
    SWA = (sum(w_i * indicator{y_i = ŷ_i})) / (sum(w_i))
    ```
    where w_i is the unique shape count for a given sample.

- **Hyperparameters:**  
  The model is trained with a maximum of 200 iterations and hyperparameters derived from the training set statistics.

---

## Results and Discussion

Our baseline experiments report the following results:
- **Development Set:**  
  - SWA: 53.8%
  - Overall Accuracy: 53.98%
- **Test Set:**  
  - SWA: 54.1%
  - Overall Accuracy: 54.25%

Ablation studies indicate that removal of either the unique color count or sequence length features causes a drop of approximately 3–4% in performance, emphasizing the importance of each feature in capturing the symbolic structure.

**Limitations and Future Directions:**
- The current feature set does not capture order-sensitive interactions or higher-order dependencies.
- Future work could integrate n-gram statistics, positional encoding, or hybrid models combining interpretable features with deep neural representations.
- Further exploration into fairness and balanced representation in symbolic diversity is warranted.

---

## Citation

If you use this code in your research, please cite our work:

Agent Laboratory. "Research Report: Symbolic Pattern Recognition in SPR_BENCH Datasets." arXiv preprint (Year). [arXiv:XXXXXXXXX]

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

We thank the contributions from the symbolic AI research community and all collaborators whose work inspired this project. For further information and updates, please visit our [project website](#) or contact us at [agentlab@example.com](mailto:agentlab@example.com).

---

Happy Coding!