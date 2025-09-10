# Enhancing Symbolic Pattern Recognition Through Hybrid Approaches

Welcome to the repository for the research project on Hybrid Symbolic Pattern Recognition (SPR). This work integrates classical TF-IDF feature extraction with explicit rule-based features and a reinforcement learning (RL) module to improve both the predictive accuracy and the interpretability of SPR tasks.

---

## Overview

Symbolic Pattern Recognition (SPR) involves identifying and interpreting explicit symbolic structures in sequences of tokens. Traditional methods like TF-IDF provide clear and interpretable features, but they often fail to capture complex domain-specific attributes. In our work, we address these limitations by:

- **Augmenting TF-IDF Representations:** Enriching the feature space with explicit symbolic diversity metrics such as unique shape count (φ(s)) and color count (ψ(s)).
- **Hybrid Model Architecture:** Combining a Random Forest classifier (statistical branch) with a reinforcement learning based rule induction module (symbolic branch), fused together using a learned gating mechanism.
- **Enhanced Interpretability:** Providing clear feature importance visualizations and rule induction analyses to explain the model’s decisions.

The final prediction in our model is computed as:

  ŷ = g(x) · f_RF(x) + (1 − g(x)) · f_RL(x)

where g(x) is determined via a sigmoid activation over latent features.

---

## Repository Contents

- **/paper/**  
  Contains the LaTeX source file of the research paper, which details the methodology, experiments, and results.

- **/code/**  
  Contains Python scripts and notebooks for:
  - Data preprocessing and feature extraction (TF-IDF, shape and color diversity).
  - Training the baseline (Logistic Regression using TF-IDF) and enhanced models (Random Forest with explicit symbolic features).
  - Implementing the reinforcement learning module for rule induction.
  - Fusion of the two branches with a learned gating mechanism.
  
- **/results/**  
  - Visualizations such as confusion matrices and feature importance plots.
  - Experimental results including Shape-Weighted Accuracy (SWA) scores.
  
- **/docs/**  
  Additional documentation, ablation study notes, and experimental logs.

- **requirements.txt**  
  A list of required Python packages including scikit-learn, numpy, matplotlib, etc.

---

## Key Features

- **Hybrid Approach:** Combines both statistical learning (via Random Forest on augmented TF-IDF features) and symbolic reasoning (via RL-based rule induction).
- **Enhanced Interpretability:** Provides tools for generating feature importance visualizations and clear rule-based explanations.
- **Domain-Specific Metrics:** Incorporates unique symbolic attributes (shape and color diversity) to capture non-linear relationships in symbolic data.
- **Reproducible Experiments:** Fixed random seeds and detailed experiment logs ensure that the results are reproducible.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- pip

### Installation

1. **Clone the Repository:**

   ```
   git clone https://github.com/YourUsername/symbolic-pattern-recognition.git
   cd symbolic-pattern-recognition
   ```

2. **Create a Virtual Environment (Optional but recommended):**

   ```
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

### Running the Code

1. **Data Preprocessing:**
   
   Execute the preprocessing script to prepare the SPR_BENCH dataset:
   
   ```
   python code/preprocess.py --data_dir data/SPR_BENCH
   ```

2. **Training the Models:**

   - **Baseline Model (Logistic Regression with TF-IDF)**
     
     ```
     python code/train_baseline.py --data_dir data/SPR_BENCH --output_dir results/baseline
     ```

   - **Enhanced Model (Random Forest with Symbolic Features + RL Module)**
     
     ```
     python code/train_enhanced.py --data_dir data/SPR_BENCH --output_dir results/enhanced
     ```

3. **Evaluating the Results:**

   Generate evaluation metrics and visualizations:

   ```
   python code/evaluate.py --model_dir results/enhanced --plot_confusion_matrix True --plot_feature_importance True
   ```

4. **Ablation Studies:**

   Run ablation studies to test the impact of each explicit feature:

   ```
   python code/ablation.py --data_dir data/SPR_BENCH --output_dir results/ablation
   ```

---

## Experimental Results

- **Baseline Performance:** Logistic Regression with TF-IDF features achieved a Shape-Weighted Accuracy (SWA) of 65.45% on the development set.
- **Enhanced Model Performance:** The hybrid approach achieved an SWA of 71.45% on development data and 70.66% on test data.
- **Interpretability:** Feature importance plots indicate that both shape variety and color variety features significantly contribute to the model's decisions.

Detailed results and visualizations can be found in the `/results/` directory.

---

## Future Work

- **Integration with Deep Neural Architectures:** Coupling the symbolic approach with transformer-based models or graph neural networks.
- **Iterative Rule Extraction:** Developing methods for continual refinement of symbolic rules.
- **Broader Domain Applications:** Extending the framework to other domains like medical diagnostics or legal analytics.
- **Enhanced Statistical Validation:** Incorporating rigorous significance testing and confidence interval estimation over multiple datasets.

---

## Contributing

Contributions to this project are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or further discussion, please reach out to the project maintainers at [agent.laboratory@example.com](mailto:agent.laboratory@example.com).

---

Happy Coding and Researching!