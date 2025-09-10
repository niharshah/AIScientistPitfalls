# Modular Atomic Predicate Decomposition for SPR

Welcome to the repository for the Modular Atomic Predicate Decomposition for Symbolic Pattern Recognition (SPR) project. This repository contains the code, data generation scripts, and experimental configurations related to our research on decomposing the SPR task into interpretable atomic predicate estimators. Our work integrates a transformer encoder with dedicated predicate modules to improve both classification accuracy and model interpretability.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture and Methods](#architecture-and-methods)
- [Experimental Setup](#experimental-setup)
- [Results and Discussion](#results-and-discussion)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

In modern tasks such as symbolic pattern recognition (SPR), standard deep-learning models—despite their high accuracy—often operate as black boxes. Our work addresses this by decomposing SPR into atomic predicate estimators that capture interpretable symbolic cues (e.g., shape-count, color-position, parity, and order). The overall decision is composed by aggregating the outputs of these predicate modules via an attentive rule aggregator. More formally, the final decision is computed as

  y = f(Σᵢ (wᵢ · pᵢ)),

where each pᵢ is an atomic predicate estimator and wᵢ is a learnable weight.

Key contributions include:
- A modular framework integrating a transformer encoder with four predicate modules.
- Empirical improvements with the augmented model (66.0% test accuracy vs. 50.5% for the baseline).
- Extensive ablation studies demonstrating that the removal of any single predicate drops accuracy to ~15.75%.
- A design that enhances interpretability and provides a clear audit trail for model decisions.

---

## Project Structure

The repository is organized as follows:

```
├── data/
│   ├── generate_data.py         # Script to generate synthetic SPR dataset
│   └── dataset_splits.csv       # Sample dataset splits information
│
├── models/
│   ├── transformer.py           # Transformer encoder module
│   ├── predicate_modules.py     # Implementation of atomic predicate estimators
│   └── aggregator.py            # Attentive rule aggregator and final classifier
│
├── experiments/
│   ├── train_baseline.py        # Training script for the vanilla transformer model
│   ├── train_augmented.py       # Training script for the augmented modular model
│   ├── ablation_study.py        # Script to perform ablation studies on predicate modules
│   └── config.yaml              # Hyperparameter and training configuration file
│
├── notebooks/
│   ├── exploratory_analysis.ipynb   # Jupyter Notebook with exploratory analysis and visualizations
│   └── results_visualization.ipynb  # Notebook for plotting experimental results and ablation outcomes
│
├── README.md                    # This readme file
├── requirements.txt             # Python package requirements
└── LICENSE                      # License information
```

---

## Installation

To set up the project on your local machine, please follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/modular-spr.git
   cd modular-spr
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Generation

Generate the synthetic dataset for SPR using the provided script:

```bash
python data/generate_data.py
```

The script creates training, development, and testing splits based on predefined symbolic rules.

### Training the Models

- **Baseline Transformer Model:**

  To train the baseline transformer model (without modular predicates):

  ```bash
  python experiments/train_baseline.py --config experiments/config.yaml
  ```

- **Augmented Modular Model:**

  To train the augmented model with atomic predicate modules:

  ```bash
  python experiments/train_augmented.py --config experiments/config.yaml
  ```

### Running Ablation Studies

To examine the impact of each predicate module on overall performance:

```bash
python experiments/ablation_study.py --config experiments/config.yaml
```

Results will show drastic performance drops (around 15.75% accuracy) when individual predicate modules are zeroed out.

### Notebooks and Visualizations

Explore the Jupyter notebooks in the `notebooks/` folder for detailed analysis and visualization of both raw experimental outcomes and ablation results.

---

## Architecture and Methods

Our system architecture comprises:

- **Transformer Encoder:** 
  - Generates contextualized embeddings for each token.
  - Utilizes a hidden dimension of 64, 2 transformer layers, and 4 attention heads.
  
- **Atomic Predicate Estimators:**
  - Four lightweight modules corresponding to shape-count, color-position, parity, and order.
  - Each module applies a linear transformation with a sigmoid activation over the transformer’s pooled output.

- **Attentive Rule Aggregator:**
  - Fuses predicate outputs using learnable weights.
  - Final prediction computed via a feed-forward network with a non-linear activation function.

- **Loss Function:**
  - Multi-task loss that combines the main binary cross-entropy loss with auxiliary losses for each predicate module:
  
    L = L_main + λ * Σᵢ L_i, where λ = 0.5.

This modular design not only improves the accuracy (with test accuracy reaching 66.0% vs. 50.5% for the baseline) but also enhances interpretability by offering insight into individual predicate contributions.

---

## Experimental Setup

Key experimental parameters include:
- **Dataset:** Synthetic sequences composed of shapes {△, □, ○, ◇} and colors {r, g, b, y}.
- **Configuration:**
  - Training: 20,000 samples
  - Development: 5,000 samples
  - Testing: 10,000 samples
- **Hyperparameters:**
  - Transformer hidden dimension: 64
  - Layers: 2; Attention Heads: 4
  - Learning Rate: 1e-3
  - Training Epochs: 5
  - Loss Balancing (λ): 0.5

The complete experimental pipeline is encapsulated in configuration files and can easily be executed on CPU-only devices to ensure reproducibility.

---

## Results and Discussion

Our experimental evaluation shows that:
- **Baseline Transformer Model:** 52.0% accuracy on the development set and 50.5% on the test set.
- **Augmented Modular Model:** 69.0% accuracy on the development set and 66.0% on the test set.
- **Ablation Studies:** Zeroing any individual predicate module leads to a drastic performance drop (≈15.75% accuracy), validating the importance of each module.

The modular approach effectively bridges the gap between high classification performance and interpretability. Detailed discussions and analysis of these results are provided in our research paper and associated notebooks.

---

## Contributing

We welcome contributions from the community! If you would like to contribute enhancements, report issues, or suggest features, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your_feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your_feature`).
5. Create a new Pull Request.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or further information, please contact:

- Agent Laboratory  
- Email: agentlab@example.com

We hope you find this repository useful and look forward to your feedback and contributions!

Happy coding!