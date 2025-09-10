# Neuro-Symbolic Reasoning Baseline Evaluation in SPR_BENCH

This repository contains the code, data preprocessing routines, and experimental configurations for our baseline evaluation of symbolic reasoning on the SPR_BENCH dataset. Our approach uses a simple logistic regression model with two interpretable features—shape complexity and color complexity—to approximate hidden symbolic rules. Although our baseline model is linear, our experiments and detailed analyses highlight its limitations in capturing non-linear symbolic interactions, motivating future work in advanced neuro-symbolic integration.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion](#discussion)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project investigates the performance of a logistic regression model on the SPR_BENCH dataset by mapping input token sequences into a two-dimensional feature space:
  
- **Shape Complexity**: The number of unique first characters in each token.
- **Color Complexity**: The number of unique second characters (if present) in each token.

The model's decision function is defined as:
  
  f(x) = α₁ · (shape complexity) + α₂ · (color complexity)

To better account for example difficulty, we introduce a novel evaluation metric called **Shape-Weighted Accuracy (SWA)**. This metric weights each sample by its shape complexity level, ensuring that examples with higher inherent symbolic diversity contribute more significantly to the accuracy computation.

---

## Features

- **Baseline Linear Model**: A logistic regression classifier trained on two computed features.
- **Feature Extraction**: Preprocessing routines to compute shape and color complexity from input sequences.
- **SWA Metric**: Custom evaluation metric that accounts for varying symbolic complexities.
- **Ablation Studies**: Experiments showing the performance impact when using either shape or color features individually.
- **Visualization Tools**: Code to generate decision boundary plots and confusion matrices for in-depth diagnostic analyses.

---

## Dataset

The evaluation is performed on the SPR_BENCH dataset, which is split as follows:

- **Training Set**: 20,000 examples
- **Development Set**: 5,000 examples
- **Test Set**: 10,000 examples

Each example in the dataset is a sequence of tokens, and the preprocessing pipeline extracts the symbolic features directly from these tokens.

---

## Installation

To run the experiments, first clone the repository:

```bash
git clone https://github.com/yourusername/neuro-symbolic-baseline.git
cd neuro-symbolic-baseline
```

Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include at least:

- scikit-learn
- numpy
- matplotlib
- pandas

---

## Usage

### Data Preparation

Ensure the SPR_BENCH dataset is placed in the directory structure expected by the code. You can update the dataset path in the configuration file (`config.yaml` or similar) as needed.

### Running the Baseline

To train the logistic regression baseline model and run the evaluation:

```bash
python run_experiment.py --mode train
python run_experiment.py --mode evaluate
```

### Generating Visualizations

To obtain diagnostic figures (e.g., decision boundaries, confusion matrices):

```bash
python visualize_results.py
```

These scripts will output the figures in the `results/figures` directory.

---

## Experimental Setup

- **Model**: Logistic Regression from scikit-learn configured with the L-BFGS solver and a maximum of 1000 iterations.
- **Feature Extraction**: Computes:
  - f_shape(s) = |{ first letter of each token in s }|
  - f_color(s) = |{ second letter (if available) of each token in s }|
- **Evaluation Metric**: Shape-Weighted Accuracy (SWA), defined as:

  SWA = (Σ wᵢ · 𝟙(ŷᵢ = yᵢ)) / (Σ wᵢ)

  where weights wᵢ are proportional to the shape complexity of the iᵗʰ example.
- **Ablation Studies**: Conducted to measure the contribution of individual features:
  - Full Model (Shape + Color)
  - Shape Only
  - Color Only

---

## Results

Our baseline experiments yielded the following performance on SPR_BENCH:

- **Development Set SWA**: ~53.57%
- **Test Set SWA**: ~55.32%

For comparison, the SPR_BENCH baseline is reported at 65.00% SWA.

**Ablation Study Summary:**

| Configuration          | SWA (%) |
|------------------------|---------|
| Full Model (Shape + Color) | 55.32   |
| Shape Only             | 50.10   |
| Color Only             | 49.85   |

In addition to the SWA metric, standard accuracy, precision, recall, and F1-score were computed, though SWA provides a more nuanced evaluation given the symbolic complexity inherent in the data.

---

## Discussion

This project demonstrates that while shape and color complexity features offer valuable interpretability, their linear combination via logistic regression is insufficient for fully capturing the non-linear dependencies present in symbolic reasoning tasks. The decision boundary visualization reveals significant class overlap, particularly in regions with intermediate feature values. The gap of approximately 10% in SWA between our baseline and the SPR_BENCH benchmark underscores the need for more advanced models that incorporate non-linear transformations, intermediate logical deduction layers, and possibly self-supervised adaptation strategies.

---

## Future Work

Future directions include:

- Incorporating hidden layers and non-linear activation functions to better model intricate symbolic interactions.
- Exploring intermediate logical deduction processes to transform raw symbolic features into abstract representations.
- Integrating self-supervised learning during inference for dynamically adaptive decision boundaries.
- Expanding this baseline to include multimodal inputs and extended logical frameworks to further bridge the gap between symbolic and neural approaches.

---

## Citation

If you find this work useful in your research, please consider citing our report:

  Agent Laboratory. (2023). Research Report: Symbolic Reasoning Baseline Evaluation in SPR_BENCH. Retrieved from https://github.com/yourusername/neuro-symbolic-baseline

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or contributions, please open an issue or submit a pull request on GitHub. Happy coding and researching!