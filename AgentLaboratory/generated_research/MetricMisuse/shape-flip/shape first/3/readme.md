# Symbolic Pattern Recognition Baseline

This repository contains the full implementation and accompanying research report for a baseline approach to Symbolic Pattern Recognition (SPR). The method integrates classical statistical techniques with an interpretable Decision Tree classifier and introduces a novel evaluation metric: the Shape-Weighted Accuracy (SWA). This metric explicitly weights samples based on the diversity of symbolic token “shapes,” offering a nuanced evaluation over standard accuracy measures.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training and Evaluation](#training-and-evaluation)
- [Experimental Setup](#experimental-setup)
- [Results and Analysis](#results-and-analysis)
- [Future Directions](#future-directions)
- [License](#license)

## Overview

Symbolic Pattern Recognition is an important research area with applications spanning natural language processing, computer vision, and robotic planning. Traditional approaches have often relied on deep neural models that sacrifice interpretability or on rule-based methods that lack flexibility. Our approach bridges this gap:
  
- **TF-IDF Feature Extraction:** Converts symbolic sequences into high-dimensional feature vectors.
- **Decision Tree Classifier:** Provides transparent decision rules based on token frequency.
- **Shape-Weighted Accuracy (SWA):** Enhances evaluation by prioritizing samples with high symbolic diversity.

The accompanying research report (included in the repository as a LaTeX document) explains the method’s theoretical foundations, experimental setup, and detailed comparative analysis against baseline methods.

## Key Features

- **Interpretable Model:** Utilizes a Decision Tree to ensure transparency in decision-making.
- **Novel Metric (SWA):** Weighs each sample by the number of unique token “shapes” for a more meaningful performance measure.
- **Reproducible Experiments:** Fixed random state for Decision Tree and complete code for data preprocessing using Python and scikit-learn.
- **Modular Design:** Easy to extend or incorporate into more advanced neuro-symbolic frameworks.

## Architecture

The primary components of the repository include:

- **Data Preprocessing Module:** Handles tokenization (whitespace-based) and TF-IDF vectorization.
- **Classifier Module:** Implements the Decision Tree classifier with configurable parameters.
- **Evaluation Module:** Computes both traditional accuracy and the SWA metric by assigning weights based on symbolic token complexity.
- **Visualization Tools:** Generates confusion matrices and histograms to analyze model performance across varying symbolic complexities.
- **Research Report:** A LaTeX paper detailing the methodology, experiments, results, and discussions.

## Installation

### Prerequisites

- Python 3.7 or later
- pip (Python package installer)

### Dependencies

Install the required packages using pip:

```bash
pip install numpy scikit-learn matplotlib
```

(Optional: Create and activate a virtual environment before installing dependencies.)

### Repository Structure

```
├── data/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── src/
│   ├── preprocess.py        # Tokenization and TF-IDF feature extraction
│   ├── train_model.py       # Training the Decision Tree classifier
│   ├── evaluate.py          # Evaluation metrics including SWA computation
│   └── visualize.py         # Tools for generating confusion matrix and histograms
├── report/
│   └── research_report.tex  # LaTeX source of the full paper
├── README.md
└── requirements.txt         # List of Python dependencies
```

## Usage

### Data Preprocessing

The raw symbolic dataset (SPR_BENCH) should be organized in CSV files within the `data` folder. Each record must include:
- A unique identifier
- A symbolic sequence (tokens separated by whitespace)
- The target label (an integer)

Run the preprocessing script to convert symbolic sequences into TF-IDF feature vectors:

```bash
python src/preprocess.py --input data/train.csv --output data/train_features.npz
```

### Training and Evaluation

Train the Decision Tree classifier using the extracted features:

```bash
python src/train_model.py --train data/train_features.npz --dev data/dev_features.npz --random_state 42
```

Evaluate the model on the test set and compute both the traditional accuracy and the SWA metric:

```bash
python src/evaluate.py --test data/test_features.npz --model saved_model.pkl
```

Visualization of the confusion matrix and token complexity histogram is generated automatically by the evaluation module, and saved as `Figure_1.png` and `Figure_2.png` respectively.

## Experimental Setup

- **Dataset:** SPR_BENCH with 20,000 training, 5,000 development, and 10,000 testing samples.
- **Features:** TF-IDF vectorization based on raw token counts.
- **Classifier:** Decision Tree with a fixed random state (42) and controlled tree depth.
- **Metric:** Shape-Weighted Accuracy (SWA) which computes:
  
  SWA = (Σ wᵢ * I(yᵢ = ŷᵢ)) / (Σ wᵢ),
  
  where wᵢ is the number of unique initial characters (shapes) in the sequence tokens.
  
- **Results:** The model achieves SWA values of 87.48% (training), 69.29% (development), and 68.14% (test), showing performance improvements over a baseline of ~65%.

## Results and Analysis

The quantitative and qualitative analysis provided in the report indicates:
- **Training Performance:** High SWA on training data, implying potential overfitting issues.
- **Generalization Gap:** A modest drop in SWA on unseen development and test samples.
- **Symbolic Complexity Impact:** Samples with higher complexity are more challenging, reinforcing the importance of the SWA metric.

Refer to the research report and generated visualizations in `report/` for further insights and detailed experiments.

## Future Directions

Future work includes:
- Extending the model to more complex ensemble methods or hybrid neuro-symbolic architectures.
- Exploring advanced tokenization strategies (e.g., sub-token decomposition).
- Integrating additional evaluation metrics for a comprehensive analysis of error distributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or contributions, please open an issue or submit a pull request. Happy coding!