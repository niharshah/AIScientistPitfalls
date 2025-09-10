# SPR_BENCH Symbolic Pattern Recognition

This repository contains the code, experiments, and documentation for our preliminary exploration of symbolic pattern recognition in the SPR_BENCH dataset. Our work focuses on evaluating a baseline logistic regression model that leverages elementary count-based features—specifically, shape complexity, color complexity, and token count—to decipher abstract token sequences with overlapping symbolic features.

The research is documented in our accompanying paper, which details the methodology, experimental setup, results, and future directions for advancing neuro-symbolic systems with more sophisticated sequential models (e.g., RNNs or Transformer-based architectures).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results & Discussion](#results--discussion)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

Symbolic pattern recognition is a challenging task, especially when abstract token sequences exhibit significant overlap in symbolic features. This repository documents our approach using a logistic regression model with features derived from:

- **Shape Complexity:** Number of distinct shape markers.
- **Color Complexity:** Number of unique color tokens.
- **Token Count:** Total number of tokens in the sequence.

To better capture content-dependent complexity, we introduce a novel evaluation metric—the **Shape-Weighted Accuracy (SWA)**. This metric assigns higher importance to sequences with a greater diversity of shape markers and is defined as:

  SWA = (Σᵢ wᵢ * 1(ŷᵢ = yᵢ)) / (Σᵢ wᵢ)

where wᵢ is the number of unique shapes in the i-th sequence and 1(·) is the indicator function.

The provided experiments yield a raw accuracy of ~54% (with SWA values very close to the raw accuracy) on the SPR_BENCH development and test splits, confirming both the potential and limitations of using simple count-based features.

---

## Repository Structure

The repository is organized as follows:

- **`data/`**  
  Contains scripts or instructions for processing the SPR_BENCH CSV files (with columns `id`, `sequence`, and `label`).

- **`src/`**  
  Source code for feature extraction, model training, and evaluation. Key files include:
  - `feature_extraction.py`: Parses input sequences and computes shape complexity, color complexity, and token count.
  - `model.py`: Implements the logistic regression classifier using scikit-learn.
  - `evaluation.py`: Contains evaluation routines including the calculation of raw accuracy and Shape-Weighted Accuracy (SWA).

- **`experiments/`**  
  Jupyter notebooks and scripts for:
  - Running ablation studies (e.g., removal of token count).
  - Visualizing performance metrics (confusion matrix, scatter plots).

- **`docs/`**  
  Contains the paper in PDF/LaTeX format and additional experiment notes.

- **`README.md`**  
  This file.

---

## Installation

To run the experiments locally, ensure you have Python (>=3.7) installed. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes packages such as:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter (optional, for running notebooks)

---

## Usage

### Data Preparation

1. Place the SPR_BENCH dataset CSV files in the `data/` directory.
2. The data should have the columns: `id`, `sequence`, and `label`.

### Running Feature Extraction & Training

To extract features and train the logistic regression model, run:

```bash
python src/model.py --data_path data/spr_bench.csv --max_iter 1000
```

This script:
- Extracts the feature vector x = [shape_complexity, color_complexity, token_count] for each sequence.
- Trains a logistic regression classifier using scikit-learn.
- Outputs predictions and evaluation metrics (raw accuracy and SWA) on the development and test splits.

### Evaluation

The evaluation script (`src/evaluation.py`) computes:
- Raw Accuracy: The percentage of correctly predicted labels.
- Shape-Weighted Accuracy (SWA): The accuracy weighted by the number of unique shape markers per sequence.

You can also run provided notebooks in the `experiments/` folder to reproduce visualizations (e.g., confusion matrices and scatter plots).

---

## Experimental Setup

The experiments were conducted using:
- A training/dev/test split of 70%/15%/15%.
- Logistic regression with a maximum of 1000 iterations.
- Feature extraction focusing on:
  - Shape complexity (number of unique shape markers).
  - Color complexity (number of unique color markers).
  - Token count (total tokens in the sequence).

The evaluation metrics (raw accuracy and SWA) along with an ablation study highlight the importance of each feature in accurately modeling symbolic sequences.

---

## Results & Discussion

The baseline model achieved:
- **Development Split:** Raw Accuracy: 53.98%, SWA: 53.82%
- **Test Split:** Raw Accuracy: 54.25%, SWA: 54.11%

Despite the inherent limitations of a simple count-based logistic regression model in capturing sequential dependencies, these results serve as a foundation for exploring more sophisticated architectures such as RNNs and Transformer-based models to improve performance.

The detailed discussion in the paper emphasizes:
- Limitations of count-based representations in capturing subtle inter-token dependencies.
- The potential benefits of integrating advanced sequential models.
- Future directions, including enhanced feature engineering and alternative evaluation metrics.

---

## Future Work

Key directions for future research include:
- **Advanced Sequential Models:** Integration of recurrent neural networks or Transformer-based architectures to capture token order and spatial relationships.
- **Enhanced Feature Engineering:** Experimenting with higher-order interactions, polynomial feature expansion, or kernel-based methods.
- **Evaluation Metrics:** Exploring alternative multitask evaluation measures to capture additional facets of symbolic complexity.
- **Data Augmentation:** Introducing noise or token rearrangement techniques to enhance robustness and generalize the model.

---

## Citation

If you use this code or dataset in your research, please cite our paper:

Agent Laboratory. (202X). "A Preliminary Exploration of Symbolic Pattern Recognition in SPR_BENCH". [arXiv reference/DOI if available].

---

## License

This repository is released under the MIT License. See the [LICENSE](LICENSE) file for additional details.

---

For any further questions or contributions, please create an issue or submit a pull request. Happy experimenting!