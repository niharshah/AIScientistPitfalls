# Neuro-Symbolic Transformer for Synthetic PolyRule Reasoning

Welcome to the repository for the Neuro-Symbolic Transformer project. This repository hosts the code, experiments, and documentation associated with our work on bridging continuous neural representations with discrete symbolic logic for complex reasoning tasks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture and Methodology](#architecture-and-methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository implements a novel neuro‑symbolic Transformer model designed for synthetic poly‑rule reasoning on L-token sequences—where each token encodes attributes such as shape and color. In our approach, a lightweight Transformer encoder is combined with a differentiable symbolic extraction layer to generate interpretable symbolic predicates. These predicates, corresponding to features like shape-count, color-position, parity, and order, are aggregated to produce a final binary decision.

Our work aims to bridge the gap between the high performance of deep learning models and the interpretability offered by traditional symbolic systems. Detailed experiments on the SPR_BENCH dataset demonstrate the model’s ability to learn and generalize complex reasoning tasks, all while providing transparent decision-making steps.

---

## Features

- **Neuro-Symbolic Architecture:** Integrates a Transformer encoder with explicit symbolic predicate extraction.
- **Interpretable Predictions:** Generates nearly discrete predicates for shape, color, parity, and order for easy inspection.
- **Differentiable Symbolic Mapping:** Uses linear heads with sigmoid activations and thresholding to map continuous features to symbolic representations.
- **Rule Verification Module:** Aggregates predicate activations using a linear rule verifier to produce final decisions.
- **Robust Training:** Demonstrates steady improvements in training loss and shape-weighted accuracy (SWA) on controlled synthetic datasets (SPR_BENCH).

---

## Architecture and Methodology

The project is structured around three key components:

1. **Embedding Layer:**  
   Each input sequence S = {s₁, s₂, …, s_L} is converted into continuous representations using token embeddings and positional embeddings:
   
   xᵢ = E_token(sᵢ) + E_pos(i)
   
2. **Transformer Encoder:**  
   The embedded sequence is processed by a lightweight Transformer encoder, which utilizes multi-head attention to capture inter-token dependencies and generate a context-rich aggregated representation.

3. **Symbolic Predicate Extraction and Rule Verification:**  
   - **Predicate Heads:** Four separate linear layers transform the aggregated vector into predicate activations:
     
     pᵢ = σ(Wᵢ x + bᵢ) for i ∈ {1,2,3,4}
     
     A threshold function converts these activations into near-binary values.
   - **Rule Verifier:** Aggregates the binary predicates via a weighted summation to yield the final decision.
     
     z = Σᵢ (wᵢ pᵢ) + b
     
   - **Loss Function:** Uses binary cross-entropy to drive learning while balancing both continuous and symbolic objectives.

Detailed discussions of our design choices, thresholding/pruning strategies, and ablation studies are included in our accompanying paper and documentation within this repository.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) (version ≥ 1.7)
- Other dependencies: numpy, matplotlib, tqdm, etc.

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/neuro-symbolic-transformer.git
   cd neuro-symbolic-transformer
   ```
2. Create and activate your virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

### Training

The main training script is implemented in `train.py`. To start training on the SPR_BENCH dataset, run:

```
python train.py --dataset SPR_BENCH --epochs 3 --batch_size 64 --lr 1e-3
```

You can customize the hyperparameters via command-line arguments.

### Evaluation

To evaluate a trained model, use the `evaluate.py` script:
   
```
python evaluate.py --model_path path_to_trained_model.pth --dataset SPR_BENCH
```

The evaluation metrics include development loss and shape-weighted accuracy (SWA).

### Visualization

Training and metric plots (loss curves, SWA trends) are generated during training and stored under the `logs/` directory. You can also run:

```
python visualize.py --log_dir logs/
```

to see an aggregated visualization of the training dynamics.

---

## Experimental Setup

### Dataset

- **SPR_BENCH:**  
  - 20,000 training instances  
  - 5,000 development instances  
  - 10,000 test instances  
  - Each instance is an L-token sequence with tokens mapped to 17 discrete indices, representing combinations of shapes and colors.

### Configuration Details

- Embedding dimension: d = 32  
- Transformer: 1 layer with 4 attention heads  
- Predicate Extraction: 4 separate linear heads with sigmoid activations and threshold τ = 0.5  
- Optimizer: Adam with learning rate = 1e-3  
- Loss Function: Binary cross-entropy

For full experimental details, please see our paper in the repository.

---

## Results

Our experiments demonstrate:

- Reduction of development loss from 0.6852 to 0.6457 over 3 epochs.
- Improvement of shape-weighted accuracy (SWA) on the development set from 55.56% to 63.85%.
- A final test set SWA of 65.18%.

Ablation studies indicate that each predicate extraction head contributes significantly—with removals leading to relative SWA drops of 2–4%. Detailed statistical and visual analyses are available in the results folder and within the paper.

---

## Future Work

Planned future directions include:

- Refinement of thresholding mechanisms (possibly adaptive thresholds).
- Exploration of non-linear aggregation strategies in the rule verifier.
- Application of the framework to more diverse, real-world datasets.
- Extension to multi-task learning scenarios to improve generalization and scalability.
- Further investigation into ethical considerations such as fairness and accountability in neuro‑symbolic reasoning.

---

## Contributing

Contributions are welcome! If you have improvements, bug fixes, or ideas for further developments, please feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out with any questions or feedback. Happy coding!