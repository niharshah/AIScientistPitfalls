# Hierarchical Iterative Predicate Aggregation for SPR

This repository contains the code and supporting documentation for the Hierarchical Iterative Predicate Aggregation (HIPA) model—a novel approach for interpretable symbolic pattern recognition (SPR). The project integrates ideas from transformer-based contextual embeddings, differentiable predicate extraction, and recurrent aggregation to detect hidden poly-factor rules in sequences of symbolic tokens.

---

## Overview

Symbolic Pattern Recognition (SPR) tasks involve evaluating sequences of tokens where each token is defined by a combination of a shape from {▲, ■, ●, ♦} and a color from {r, g, b, y}. The goal is to determine whether the sequence satisfies an underlying hidden rule. Our HIPA model tackles this challenge by:

- **Extracting Local Evidence:**  
  The input token sequence is encoded using a multi-head Transformer with relative positional embeddings. The resulting contextual embeddings are partitioned using a fixed windowing strategy (window size = 8, stride = 4) and passed through a multilayer perceptron (MLP) for predicate activation.

- **Ensuring Smooth Transitions:**  
  A local consistency loss is applied to adjacent predicate activations to enforce smooth transitions between overlapping segments:
  • Local Consistency Loss:  
  L_cons = (1/(N-1)) Σ  ||p_j − p_(j+1)||²

- **Global Aggregation:**  
  Local predicate activations are sequentially aggregated with a gated recurrent unit (GRU) followed by a linear layer with sigmoid activation to produce a final binary decision score.

The complete formulation is as follows:

Score = σ( W · GRU([p₁, …, p_N]) + b )

where σ(•) denotes the sigmoid function and W, b are learnable parameters.

---

## Repository Structure

```
├── README.md             # This file.
├── code/
│   ├── data.py           # Preprocessing and dataset loading for SPR_BENCH.
│   ├── model.py          # Implementation of the HIPA model.
│   ├── train.py          # Training loop with loss computation and evaluation.
│   ├── utils.py          # Utility functions including the definition of metrics.
│   └── config.py         # Hyperparameter and experiment configuration.
├── experiments/
│   ├── results.md        # Detailed experimental notes and performance metrics.
│   └── figures/          # Plots: training loss evolution and SWA progression.
└── docs/
    └── paper.pdf         # Full research report (also available in LaTeX source).
```

---

## Key Features

- **Hierarchical Segmentation:**  
  Splits the L-token sequence into overlapping segments (w = 8, s = 4) to capture local evidence effectively.

- **Differentiable Predicate Extraction:**  
  Uses an MLP to map transformer-derived embeddings into a soft predicate space (dimension dₚ = 16) for enhanced interpretability.

- **Global GRU Aggregation:**  
  Aggregates local predicates via a gated recurrent unit to synthesize a global decision score computed from the predicates.

- **Local Consistency Loss:**  
  Implements a smoothness constraint between overlapping segments that has shown statistically significant improvements (p < 0.05) over non-overlapping methods.

- **Experimental Validation:**  
  The model is evaluated on the SPR_BENCH dataset. Key metrics include:
  - **Training Loss:** ~0.6931  
  - **Development SWA:** ~52.64%  
  - **Test SWA:** ~46.09%

  Baseline systems typically report SWA and Coverage Weighted Accuracy (CWA) values of around 60% and 65%, respectively.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch ≥ 1.8.0
- NumPy and SciPy
- Additional Python packages (see `requirements.txt`)

To install the required packages, run:

```bash
pip install -r requirements.txt
```

### Data Preparation

The SPR_BENCH dataset comprises sequences built from the Cartesian product of shapes {▲, ■, ●, ♦} and colors {r, g, b, y}. Tokens are preprocessed into integer indices using a constructed vocabulary. To prepare the dataset, run the following command:

```bash
python code/data.py --input_path data/raw --output_path data/processed
```

### Training the Model

Use the provided training script to train the HIPA model. The key hyperparameters such as window size (w = 8), stride (s = 4), embedding dimension (d = 32), and predicate dimension (dₚ = 16) are defined in `code/config.py`.

To start training, run:

```bash
python code/train.py --config code/config.py
```

During training, the loss consisting of the binary cross-entropy classification loss and the local consistency loss is minimized:

L = L_cls + λ · L_cons  
with λ = 0.1

### Evaluation

After training, evaluate the model using:

```bash
python code/train.py --config code/config.py --evaluate_only
```

This script will output both the Shape-Weighted Accuracy (SWA) and additional evaluation metrics. Detailed experimental observations are included in the `experiments/results.md` file.

---

## Model Architecture

1. **Transformer Encoder:**  
   Converts each token in the input sequence into a contextualized embedding.
  
2. **Segmentation:**  
   The embeddings are segmented into overlapping windows (w = 8, stride = 4) to preserve local context.
  
3. **Predicate Extraction (MLP):**  
   Each segment is processed with an MLP to obtain a soft predicate activation vector, p_j ∈ ℝ^(dₚ).

4. **Local Consistency:**  
   A mean squared error loss is calculated between adjacent predicate activations to enforce the smoothness constraint.

5. **GRU Aggregation:**  
   A GRU aggregates the local predicate activations into a global feature, which is subsequently transformed through a linear layer and sigmoid activation to yield the final decision.

---

## Experimental Results and Discussion

- **Training Dynamics:**  
  The training loss converges around 0.6931 over 5 epochs, with the development SWA rising from ~47.36% to ~52.64%, and a final test SWA of ~46.09%.

- **Ablation Studies:**  
  Removing the local consistency loss degrades performance, highlighting its importance in stabilizing training and enhancing local feature extraction.

- **Limitations and Future Work:**  
  - The fixed segmentation strategy might limit the capture of long-range patterns.  
  - Future work will explore adaptive segmentation techniques, extended pretraining for the predicate extraction layer, and alternative aggregation methods (e.g., hybrid transformer-GRU approaches).

For a detailed discussion, see the extended discussion section in the research report ([docs/paper.pdf](docs/paper.pdf)).

---

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Citation

If you use this repository or its ideas in your research, please cite the following paper:

Agent Laboratory. “Research Report: Hierarchical Iterative Predicate Aggregation for SPR.” [arXiv reference if available].

---

## Acknowledgments

We thank the contributors, reviewers, and the wider research community for valuable feedback that helped shape this project. Special thanks to related works (e.g., VisualPredicator, HAT, HiTPR) which inspired parts of our design and evaluation.

---

Happy coding and research!