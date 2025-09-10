# Symbolic Pattern Recognition with LSTM-based Classifiers

Welcome to the repository for our research work on Symbolic Pattern Recognition (SPR) using an LSTM-based classifier. This project presents a lightweight yet effective neural architecture that maps variable-length symbolic sequences to predictive labels by integrating embedding layers, recurrent networks, and pooling operations.

---

## Overview

The goal of this project is to address the challenge of uncovering hidden logic in symbolic sequences—a problem with wide applications in natural language processing, automated theorem proving, and beyond. Our approach leverages the following transformation:

  h = mean(LSTM(E(x)))  
  ŷ = argmax(W * h + b)

Where:  
- x is a symbolic sequence  
- E(·) maps tokens to a continuous embedding space  
- LSTM(·) processes the embedded sequence  
- Mean pooling aggregates the sequential outputs  
- A final linear layer produces the class scores

This readme provides an introduction to the project, instructions for replication and experimentation, and an overview of design decisions and experimental findings.

---

## Repository Structure

```
.
├── docs
│   └── paper.pdf                # The full research paper (also available in LaTeX)
├── figures
│   ├── Figure_1.png             # Training Loss per Epoch
│   └── Figure_2.png             # Development Accuracy (SWA) per Epoch
├── experiments
│   ├── run_experiment.py        # Main training and evaluation script
│   └── preprocessing.py         # Tokenization and sequence padding utilities
├── models
│   └── lstm_classifier.py       # Implementation of the LSTM-based classifier
├── data
│   └── SPR_BENCH/               # Dataset partitioned into training, development, and test splits
├── requirements.txt             # Python dependencies file
└── README.md                    # This file
```

---

## Key Features

- **LSTM-Based Architecture:**  
  Leverages a single-layer LSTM with mean pooling to capture both local and global symbolic dependencies.

- **Embedding Layer:**  
  Maps each token from a discrete vocabulary into a 32-dimensional continuous vector.

- **Reproducible Experiments:**  
  The repository contains all necessary code to reproduce experiments on the SPR_BENCH dataset, including preprocessing, training under CPU-only constraints, and evaluation using the Shape-Weighted Accuracy (SWA) metric.

- **Empirical Evaluation:**  
  Experimental results show training loss reduction from 0.6689 to 0.6031 and development SWA improvement from 75.00% to 78.00% over two epochs. Although our test set SWA of 66.50% does not match state-of-the-art baselines, our work provides a solid baseline and clear avenues for future improvements.

---

## Installation

1. **Clone the Repository:**

  `git clone https://github.com/yourusername/symbolic-pattern-recognition.git`

2. **Navigate to the Repository Directory:**

  `cd symbolic-pattern-recognition`

3. **Create a Python Virtual Environment (Optional but Recommended):**

  `python3 -m venv venv`  
  `source venv/bin/activate`

4. **Install Dependencies:**

  `pip install -r requirements.txt`

*Note: The code has been tested using Python 3.8+.*

---

## Usage

### Data Preparation

Place the SPR_BENCH dataset within the `data/SPR_BENCH` directory. The dataset should include three splits:
- `train`: 500 examples
- `dev`: 200 examples
- `test`: 200 examples

The preprocessing script in `experiments/preprocessing.py` tokenizes the input sequences, constructs the vocabulary (based on the training set), and pads the sequences to enable uniform batch processing.

### Training and Evaluation

Run the main training script:
```
python experiments/run_experiment.py
```
This script will:
- Load and preprocess the data.
- Initialize the LSTM-based model defined in `models/lstm_classifier.py`.
- Train the model for two epochs using the Adam optimizer (learning rate: 1e-3, batch size: 32).
- Record training loss (0.6689 -> 0.6031) and SWA improvements on the development set (75.00% -> 78.00%).
- Evaluate the final model on the test set (observed SWA: 66.50%).

### Visualizations

The figures in the `figures/` directory illustrate:
- **Figure 1:** Training loss per epoch.
- **Figure 2:** Development accuracy (SWA) per epoch.

You can update or generate additional plots by modifying the relevant visualization code in the experiment scripts.

---

## Experimental Details

- **Model Hyperparameters:**
  - Embedding Dimension: 32
  - LSTM Hidden Dimension: 64
  - Batch Size: 32
  - Epochs: 2

- **Loss Function:**
  - Cross-entropy loss is minimized over all training examples.

- **Optimizer:**
  - Adam with an initial learning rate of 1×10⁻³.

- **Evaluation Metric:**
  - Shape-Weighted Accuracy (SWA) is used to emphasize the role of structural components in symbolic sequences.

For further details on the methodology, experimental setup, and broader discussion of results, please refer to the research paper located in the `docs/` directory.

---

## Future Work

The current model serves as a baseline for SPR tasks. Future improvements include:
- Extending the training duration beyond two epochs.
- Experimenting with deeper LSTM networks or incorporating bidirectionality.
- Exploring alternative architectures (e.g., Transformers) and attention mechanisms for refined feature extraction.
- Increasing dataset diversity and size, and improving the tokenization strategy.
- Integrating hybrid neural-symbolic approaches to enhance both performance and interpretability.

---

## References

1. arXiv:2503.04900v1  
2. arXiv:2203.00162v3  
3. arXiv:1709.01490v2  
4. arXiv:2501.00296v3  
5. arXiv:2505.23833v1  
6. arXiv:2410.23156v2  
7. arXiv:2505.06745v1  
8. arXiv:2506.14373v2  

For a full list of references, see the paper in `docs/paper.pdf`.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Thank you for checking out our work on Symbolic Pattern Recognition with LSTM-based classifiers. We hope this repository serves as a valuable resource for researchers interested in neural-symbolic integration. Happy experimenting!