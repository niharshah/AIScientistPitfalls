# PolyRule Reasoning Transformer (PRT)

Welcome to the GitHub repository for the PolyRule Reasoning Transformer (PRT), a neuro–symbolic hybrid model designed for Symbolic Pattern Recognition (SPR). This project integrates a lightweight Transformer encoder with an explicit, differentiable rule induction module for improved performance and enhanced interpretability on symbolic datasets.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Experimental Results](#experimental-results)
- [Visualization & Ablation Studies](#visualization--ablation-studies)
- [Discussion & Future Work](#discussion--future-work)
- [License](#license)

---

## Overview

Symbolic Pattern Recognition (SPR) tasks involve processing sequences where each token is a composite symbol (e.g., an abstract shape paired with a color). Traditional deep learning models achieve high predictive performance but often work as black boxes. The PRT model addresses this challenge by explicitly incorporating a rule induction module within a Transformer-based architecture. The key contributions of this project include:

- **Hybrid Approach:** Combining a two-layer Transformer encoder with a differentiable rule induction module.
- **Explicit Rule Induction:** Computing atomic predicate scores (Shape-Count, Color-Position, Parity, Order) that allow for fine-grained interpretability.
- **Differentiable AND-like Aggregation:** Obtaining the final prediction through the product of sigmoid-activated predicate scores.
- **Empirical Validation:** Experiments on a synthetic SPR dataset and the SPR_BENCH dataset demonstrating improved accuracy and Shape-Weighted Accuracy (SWA).

---

## Model Architecture

The PRT model is composed of the following key components:

1. **Embedding & Positional Encoding:**  
   - Tokens are embedded into a 32-dimensional continuous vector space.
   - Sinusoidal positional encodings are added to capture token order.

2. **Transformer Encoder:**  
   - A two-layer Transformer with 2 attention heads per layer is used.
   - It captures long-range dependencies and token interactions via the self-attention mechanism.

3. **Differentiable Rule Induction Module:**  
   - Computes predicate scores for four symbolic features:
     - **Shape-Count:** Diversity of shapes.
     - **Color-Position:** Positional arrangement of colors.
     - **Parity:** Even or odd occurrence of features.
     - **Order:** Sequential order based on predefined rules.
   - Each predicate is processed with a feed-forward network followed by a sigmoid activation.
   - Final output probability is obtained as:  
     p = σ(s₁) · σ(s₂) · σ(s₃) · σ(s₄)

4. **Baseline Model:**  
   - A standard Transformer-based classifier without the rule induction module, used for comparative evaluation.

---

## Datasets

Two datasets are employed for evaluation:

1. **Synthetic Dataset:**  
   - Contains 100 samples with 10-token sequences.
   - Each token carries symbolic attributes (shape and color).
   - Used to rigorously test the model's ability to capture poly–factor rules in a controlled environment.
   - Split: 80% training, 20% testing.

2. **SPR_BENCH Dataset:**  
   - A more diverse, real-world dataset with 20,000 training samples, 5,000 development samples, and 10,000 testing samples.
   - Each sample comprises a unique token sequence with an associated binary label defined by hidden symbolic rules.

---

## Installation

Ensure you have [Python](https://www.python.org/downloads/) installed (version 3.7 or above) along with [PyTorch](https://pytorch.org/). Then, clone the repository and install required packages:

```bash
# Clone the repository
git clone https://github.com/your-username/PolyRule-Reasoning-Transformer.git
cd PolyRule-Reasoning-Transformer

# (Optional) Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

*Note: The `requirements.txt` file contains all necessary dependencies (e.g., torch, numpy, matplotlib).*

---

## Usage

To run the experiments and validate the model:

1. **Training on the Synthetic Dataset:**

   ```bash
   python train.py --dataset synthetic --epochs 20 --lr 0.005
   ```

2. **Training on the SPR_BENCH Dataset (Baseline Transformer):**

   ```bash
   python train.py --dataset spr_bench --use_rule_module False --epochs 20 --lr 0.005
   ```

3. **Visualization & Analysis:**

   Generate heatmaps of predicate activations and perform ablation studies by running the visualization script:

   ```bash
   python visualize.py --sample_index 5
   python ablation.py
   ```

*Command-line options can be adjusted as needed. Check the script’s help for more details:*

```bash
python train.py --help
```

---

## Training Details

- **Hyperparameters:**
  - Embedding Dimension: 32
  - Number of Transformer Layers: 2
  - Attention Heads per Layer: 2
  - Hidden Dimension: 32
  - Learning Rate: 0.005
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy
  - Number of Epochs: 20

- **Data Preprocessing:**
  - Tokenization is performed using whitespace-based splitting.
  - Each sequence is converted into a fixed-length tensor (10 tokens per sequence).

- **Evaluation Metrics:**
  - **Accuracy:** Fraction of correctly classified samples.
  - **Shape-Weighted Accuracy (SWA):** Weights each sample by the number of unique shapes:
    SWA = (Σ_i [wᵢ · I{yᵢ = ŷᵢ}]) / (Σ_i wᵢ)

---

## Experimental Results

### Synthetic Dataset
- **PRT Model:**
  - Test Accuracy: 1.0000
  - SWA: 1.0000
- The model converges quickly with training loss decreasing from 2.8278 (epoch 1) to 0.0150 (epoch 20).

### SPR_BENCH Dataset (Baseline Transformer)
- **Baseline Transformer (without rule induction):**
  - Development Accuracy: 72.76%
  - SWA: 69.25%

Additional ablation studies further confirm that the removal of the rule induction module leads to performance degradation, highlighting its importance for capturing poly–factor rules in symbolic data.

---

## Visualization & Ablation Studies

- **Heatmaps:**  
  Visualize predicate activations for individual test samples to gain insights into the rule induction process.
- **Ablation Studies:**  
  Evaluate the impact of excluding specific predicates or the entire rule induction module. These experiments underscore the importance of integrating symbolic reasoning for enhanced performance and interpretability.

Run the provided scripts (`visualize.py` and `ablation.py`) to reproduce the analysis.

---

## Discussion & Future Work

This project demonstrates that combining symbolic reasoning with deep learning via the PRT model leads to both:
- **Superior Performance:** Particularly in controlled synthetic settings.
- **Enhanced Interpretability:** Through explicit predicate decomposition and rule aggregation.

**Future Directions:**
- Extend the framework to handle longer sequences and more complex, noisy real-world symbolic datasets.
- Explore alternative aggregation mechanisms (e.g., additive or hybrid operators) for predicate scores.
- Investigate hierarchical representations for multi-scale reasoning.
- Integrate pre-trained language models or transfer learning methods to improve generalization.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

Feel free to open issues or contribute to the project. Happy coding!

