# Symbolic Pattern Recognition Baseline

Welcome to the Symbolic Pattern Recognition (SPR) Baseline repository! This project presents a transparent, interpretable baseline for SPR tasks by combining explicit numeric feature extraction with elementary symbolic rule induction. The repository is built around a simple logistic regression model that leverages carefully engineered features such as shape complexity, color complexity, and token count to classify symbolic sequences.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion and Future Work](#discussion-and-future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project explores a baseline approach to Symbolic Pattern Recognition (SPR) with the following main components:

1. **Feature Extraction**: For each symbolic sequence, three features are computed:
   - **Shape Complexity**: The count of unique shape glyphs.
   - **Color Complexity**: The count of unique color glyphs.
   - **Token Count**: The total number of tokens in the sequence.

2. **Classification Model**: A logistic regression model is trained on the extracted feature vectors. The decision boundary is modeled as:
   
   log(P(y=1|x) / (1-P(y=1|x))) = β₀ + β₁x₁ + β₂x₂ + β₃x₃

3. **Novel Evaluation Metric**: The Shape-Weighted Accuracy (SWA) metric is introduced to weigh classification performance based on the symbolic richness of the samples. It is defined as:
   
   SWA = (Σᵢ wᵢ · I(yᵢ = ŷᵢ)) / (Σᵢ wᵢ)
   
   where wᵢ is determined by the number of unique shape glyphs.

4. **Symbolic Rule Induction**: An elementary rule induction process is implemented that postulates candidate symbolic rules in the form:
   
   {s₁, s₂, ..., sₖ} → y
   
   Rules are validated using an entailment score function with an empirical threshold.

---

## Background

The SPR task involves deciphering abstract rules that govern symbolic sequences. Traditionally rooted in classical pattern recognition and symbolic reasoning, our approach emphasizes interpretability and robust evaluation. By explicitly extracting and evaluating symbolic features, the project provides a clear and reproducible baseline which can be built upon with more advanced neuro-symbolic architectures in future work.

---

## Key Features

- **Explicit Feature Extraction**: Quantifies symbolic diversity using shape, color, and token-based metrics.
- **Interpretable Classification**: Uses logistic regression to maintain a transparent decision process.
- **Novel Evaluation Metric (SWA)**: Weighs predictions by symbolic richness, highlighting the importance of shape complexity.
- **Ablation Studies**: Assesses the impact of each feature on overall predictive performance.
- **Visualizations**: (Notebooks) Include confusion matrices and scatter plots to correlate shape complexity with prediction correctness.

---

## Repository Structure

    ├── data/
    │   └── SPR_BENCH/         # Contains the training, development, and test splits
    ├── notebooks/
    │   ├── exploratory_analysis.ipynb  # Data exploration and visualization scripts
    │   └── feature_extraction.ipynb      # Code for feature extraction and ablation studies
    ├── src/
    │   ├── model.py            # Logistic regression model and training routines
    │   ├── feature_engineering.py   # Scripts to compute shape, color, and token counts
    │   └── rule_induction.py   # Implementation of symbolic rule induction process
    ├── results/
    │   └── figures/            # Confusion matrices, scatter plots, and other visualizations
    ├── README.md               # This readme file
    └── requirements.txt        # Python dependencies

---

## Installation

To set up the repository locally, please follow these steps:

1. **Clone the Repository:**

   git clone https://github.com/yourusername/spr-baseline.git  
   cd spr-baseline

2. **Create a Virtual Environment:**

   On Linux/macOS:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

---

## Usage

1. **Data Preparation:**

   Ensure the SPR_BENCH dataset is available under the `data/SPR_BENCH/` directory. The dataset should be split into training, development, and test samples as described in the paper.

2. **Feature Extraction:**

   Run the feature extraction script to compute shape, color, and token features:
   
   ```
   python src/feature_engineering.py --input data/SPR_BENCH --output data/features.pkl
   ```

3. **Model Training and Evaluation:**

   To train the logistic regression model and evaluate using the SWA metric:
   
   ```
   python src/model.py --features data/features.pkl --output results/model_output.pkl
   ```

4. **Visualize Results:**

   Open the Jupyter notebooks in `notebooks/` to review exploratory data analysis, confusion matrices, and scatter plots:
   
   ```
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

---

## Experimental Setup

- **Dataset Splits:**  
  - Training: 20,000 samples  
  - Development: 5,000 samples  
  - Test: 10,000 samples  

- **Hyperparameters:**
  - Maximum Iterations: 1000
  - Regularization Strength (C): 1.0
  - Random Seed: 42
  - Entailment Threshold (ε): 0.9

- **Evaluation Metric:**  
  Shape-Weighted Accuracy (SWA) is used to assess performance, with experimental SWA scores of ~53.82% on development and ~54.11% on test sets.

---

## Results

Our baseline logistic regression model demonstrates that even simple statistical approaches can capture meaningful symbolic features. Key findings include:

- Consistent SWA performance across development and test sets.
- Shape complexity is a critical feature, with ablation studies showing a drop of approximately 7% in SWA when omitted.
- Visual analyses (confusion matrices and scatter plots) further confirm the correlation between enhanced symbolic diversity and prediction accuracy.

Full details of the experiments are provided in the paper and accompanying notebooks.

---

## Discussion and Future Work

While the baseline achieved reasonable performance on the SPR task, several opportunities for improvement exist:

- **Enhanced Feature Extraction:** Leveraging state-of-the-art techniques (e.g., attention-based models) to extract richer symbolic representations.
- **Integration with Neuro-Symbolic Models:** Combining deep learning approaches with explicit rule induction could capture non-linear symbolic interactions more effectively.
- **Refined Evaluation Metrics:** Incorporating additional fairness and class-weighted measures for a broader understanding of model performance.
- **Exploration of Non-Linear Models:** Experimenting with more complex models that might better capture the intricacies of symbolic relationships.

This repository serves as a starting point for researchers and practitioners interested in developing more robust and interpretable SPR systems.

---

## Contributing

We welcome contributions to enhance the project! If you have suggestions, bug fixes, or additional features, please:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request detailing your changes.

For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding and thank you for exploring the Symbolic Pattern Recognition Baseline project!