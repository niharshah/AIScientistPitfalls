# Symbolic Pattern Recognition Baseline

This repository contains the implementation and experimental framework described in the research report “Advances in Symbolic Pattern Recognition.” The work investigates a bag-of-shapes representation for transforming symbolic sequences into frequency-based feature vectors (bag-of-shapes) and classifying them using a simple feed-forward neural network with one hidden layer. Although our baseline model is computationally efficient and interpretable, the experiments reveal a performance gap relative to state-of-the-art methods, motivating further research into richer sequential and symbolic features.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project focuses on the challenging task of Symbolic Pattern Recognition (SPR). The key contributions of the work are:

- **Baseline Model:** A feed-forward neural network using a bag-of-shapes representation.
- **Feature Extraction:** Sequence tokens are mapped into a frequency count vector where each component corresponds to the occurrence of a particular shape.
- **Model Architecture:** 
  - An input layer with dimension equal to the number of unique shape tokens.
  - One hidden layer with 16 units and ReLU activation.
  - An output layer with 2 units (for binary classification) using the softmax function.
- **Evaluation Metrics:** Standard Test Accuracy and Shape-Weighted Accuracy (SWA). Both metrics yielded 61.84% on the test set.
- **Discussion & Future Directions:** A detailed analysis of the limitations of a frequency-based approach and suggestions for integrating sequential, contextual, and symbolic reasoning features.

For full details, please see the accompanying research report in the repository.

---

## Repository Structure

The repository is organized as follows:

```
├── data/
│   └── SPR_BENCH/                   # Dataset directory (20,000 training, 5,000 dev, 10,000 test samples)
├── models/
│   └── spr_baseline.py              # Implementation of the bag-of-shapes baseline model
├── notebooks/
│   └── exploratory_analysis.ipynb   # Notebook with experiments and visualizations (loss curve, SWA comparison)
├── figures/
│   ├── Figure_1.png                 # Training loss curve visualization
│   └── Figure_2.png                 # SWA comparison bar chart
├── paper/
│   └── report.pdf                   # Research report (generated from the provided LaTeX source)
├── README.md
└── requirements.txt                 # List of Python dependencies
```

---

## Installation and Dependencies

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/symbolic-pattern-recognition-baseline.git
   cd symbolic-pattern-recognition-baseline
   ```

2. **Create a virtual environment (optional but recommended):**

   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

   The repository requires standard Python packages such as:
   - NumPy
   - PyTorch or TensorFlow (depending on your implementation)
   - Matplotlib (for visualizations)
   - Pandas
   - Jupyter (if using the notebooks)

---

## Usage

### Running the Baseline Model

To train and evaluate the bag-of-shapes baseline model on the SPR_BENCH dataset, run:

```
python models/spr_baseline.py --data_dir ./data/SPR_BENCH --epochs 20 --lr 0.01
```

The script performs the following steps:
- Transforms each symbolic sequence into a bag-of-shapes vector.
- Initializes and trains a shallow feed-forward neural network.
- Logs training loss and evaluation metrics.
- Saves model checkpoints and the final performance metrics.

### Visualizing Results

For exploratory analysis and visualizations, launch the Jupyter Notebook:

```
jupyter notebook notebooks/exploratory_analysis.ipynb
```

This notebook provides visualizations similar to those reported:
- Training loss curve (Figure_1.png)
- SWA comparison bar chart (Figure_2.png)

### Viewing the Research Report

The research report is available in the `/paper` directory as a PDF generated from the LaTeX source. Open `paper/report.pdf` for a detailed description of the methodology, experimental results, discussions, and future research directions.

---

## Experimental Setup

- **Dataset:** SPR_BENCH (20,000 training samples, 5,000 development samples, 10,000 test samples)
- **Feature Extraction:** Bag-of-shapes representation where each input sequence is converted to a vector of token counts.
- **Model Architecture:**
  - Input Layer: Dimensionality equal to the number of unique shape tokens.
  - Hidden Layer: 16 units with ReLU activation.
  - Output Layer: 2 units with softmax for binary classification.
- **Training Details:**
  - Optimizer: Adam with a learning rate of 0.01.
  - Epochs: 20.
  - Loss Function: Cross-entropy loss.
  - Observed training loss reduction from 0.7351 (epoch 1) to 0.5420 (epoch 20), and developmental accuracy peaked at 79.56%.
- **Evaluation Metrics:**
  - Standard Test Accuracy: 61.84%
  - Shape-Weighted Accuracy (SWA): 61.84%

---

## Results

The baseline achieved:
- **Standard Test Accuracy:** 61.84%
- **Shape-Weighted Accuracy (SWA):** 61.84%

The identical values for both metrics indicate that the bag-of-shapes model, while simple and efficient, does not capture critical token-level nuances. Comparisons against state-of-the-art methods (approximately 75% SWA) underscore the need for more advanced feature extraction techniques (e.g., sequential modeling, attention mechanisms).

Visualizations of the training loss and SWA comparison can be found in the `/figures` folder.

---

## Future Work

Several avenues for future research are highlighted in this project:
- **Enhancing Feature Representations:** Integrate sequential or positional information to complement the bag-of-shapes abstraction.
- **Advanced Architectures:** Experiment with deeper networks, attention mechanisms, transformer-based models, or graph-based embeddings.
- **Regularization Improvements:** Investigate dropout, batch normalization, and robust hyperparameter tuning to address overfitting.
- **Data Augmentation:** Explore synthetic data generation to better capture the diversity of token sequences.
- **Hybrid Models:** Combine statistical aggregation with explicit symbolic reasoning for improved SPR performance.

These directions aim to bridge the performance gap and exploit the richness of symbolic data more effectively.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to open issues or submit pull requests if you have suggestions, improvements, or bug fixes!