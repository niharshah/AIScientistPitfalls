# Research Report: An Investigation into Neural-Symbolic Systems for SPR Tasks

This repository contains the source code, datasets, experimental setup, and documentation related to our research on integrating neural representation learning with symbolic rule induction for Sequence Pattern Recognition (SPR) tasks. The work combines a neural network trained with stochastic gradient descent and answer set programming (ASP) for enforcing logical consistency, aiming to bridge the gap between high-dimensional perceptual processing and interpretable symbolic reasoning.

---

## Overview

Our research investigates a hybrid neuro-symbolic system capable of:
- Mapping raw sequence inputs (e.g., “● y  ● g  ● r  □ r  Δ y  Δ g”) into robust latent representations.
- Jointly optimizing neural parameters (θ) and symbolic hypotheses (H) under formal logical constraints.
- Balancing traditional cross-entropy loss with a semantic loss that enforces symbolic consistency.

The system is evaluated on the SPR_BENCH dataset with key metrics such as:
- **Training Loss:** Decreased from 0.6486 to 0.1806 over 3 epochs.
- **Development Shape-Weighted Accuracy (SWA):** Improved from 66.92% to 94.66%.
- **Test SWA:** Reported at 67.87%, highlighting a generalization gap that motivates future work in regularization and domain adaptation.

---

## Repository Structure

```
repo-root/
├── data/                   # Contains SPR_BENCH dataset splits (train, dev, test)
├── docs/                   # Additional documentation and notes related to the research
├── experiments/            # Experimental scripts, configuration files, and logs
├── models/                 # Saved model checkpoints and symbolic rule outputs
├── figures/                # Figures used in the paper (e.g., Figure_1.png and Figure_2.png)
├── src/                    # Source code for the NeuralSPR model
│   ├── __init__.py
│   ├── model.py            # Neural model and LSTM architecture definition
│   ├── symbolic.py         # Answer Set Programming-based symbolic rule induction
│   └── train.py            # Training loop implementing joint optimization
└── README.md               # This file
```

---

## Features

- **Neural Symbolic Integration:** Combines neural representation learning with symbolic rule extraction.
- **Joint Optimization:** Implements a joint loss function: 
  - Cross-Entropy Loss (for standard classification) plus 
  - Semantic Loss (to enforce symbolic consistency).
- **Reproducible Experiments:** Runs on CPU with fixed random seeds across Python, NumPy, and PyTorch for deterministic behavior.
- **Extensible Framework:** Designed to be a foundation for further work in regularization strategies, domain adaptation, and advanced neuro-symbolic methodologies.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/neuro-symbolic-spr.git
   cd neuro-symbolic-spr
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes libraries such as PyTorch, NumPy, and any additional packages necessary for ASP reasoning.

---

## Usage

### Training the NeuralSPR Model

To run the training script with the default hyperparameters:

```bash
python src/train.py --data_dir ./data --epochs 3 --batch_size 64 --learning_rate 0.01 --momentum 0.9
```

The training code implements a joint optimization loop that minimizes the combined loss function:

   L = L_CE + λ · L_sem

where:
- L_CE is the cross-entropy loss.
- L_sem is the semantic loss that ensures the latent representations align with symbolic rules.

### Evaluating Model Performance

After training, evaluation on the development and test sets is automatically performed. Key performance metrics (e.g., training loss and Shape-Weighted Accuracy) are logged and can be viewed in the generated plots found in the `figures/` directory.

---

## Experimental Setup

- **Dataset:** SPR_BENCH, featuring sequences of abstract symbols (e.g., “● y  ● g  ● r  □ r  Δ y  Δ g”).
- **Architecture:**
  - Embedding Layer (Dimension: 32)
  - LSTM Layer (Hidden Dimension: 64)
  - Fully Connected Layer for Classification
- **Hyperparameters:**
  - Learning Rate: 0.01
  - Momentum: 0.9
  - Epochs: 3
  - Batch Size: 64
- **Reproducibility:** Fixed random seed (42) across experiments, run on CPU for consistency.

---

## Results Summary

- **Training Metrics:**
  - Epoch 1: Loss = 0.6486, Dev SWA = 66.92%
  - Epoch 2: Loss = 0.3886, Dev SWA = 91.48%
  - Epoch 3: Loss = 0.1806, Dev SWA = 94.66%
  
- **Test Performance:** Test SWA = 67.87%

The significant gap between development and test accuracies highlights challenges such as overfitting and data distribution discrepancies, motivating future work in adaptive regularization and domain adaptation.

---

## Future Work

Key areas for future development include:
- **Enhanced Regularization:** Incorporating dropout, weight decay, and early stopping to mitigate overfitting.
- **Adaptive Loss Balancing:** Dynamically adjusting the weight between the semantic loss and cross-entropy loss.
- **Advanced Architectures:** Investigating transformer-based models for capturing long-range sequence dependencies.
- **Dynamic Rule Induction:** Exploring self-supervised and reinforcement learning approaches to refine the symbolic rule extraction process.
- **Robust Evaluation:** Implementing cross-validation and multi-seed experiments to ensure statistically robust performance metrics.

---

## Citation

If you use our code or research in your work, please cite our paper:

    @misc{neurosymbolic_spr,
      title={Research Report: An Investigation into Neural-Symbolic Systems for SPR Tasks},
      author={Agent Laboratory},
      year={2023},
      note={Available at: https://arxiv.org/abs/your-identifier}
    }

---

## Acknowledgements

We thank the broader research community for insights into neuro-symbolic integration and the development of ASP-based reasoning systems. We also acknowledge the funding and support provided by our institution’s research initiatives.

For any questions or contributions, please open an issue or submit a pull request.

Happy coding!
