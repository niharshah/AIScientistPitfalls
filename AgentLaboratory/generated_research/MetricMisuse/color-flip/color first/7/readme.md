# Neuro-Symbolic Integration for Symbolic Pattern Recognition

This repository contains the code and documentation for our research on a novel neuro-symbolic hybrid model for Symbolic Pattern Recognition (SPR). In this project, we integrate a transformer-based contextual encoder with an explicit symbolic rule extraction module to capture both discreet symbolic cues and rich deep representations. The final decision is computed as an average of the transformer (base) logit and the logic-based (symbolic) logit.

## Overview

Symbolic Pattern Recognition poses unique challenges where subtle rule violations in sequences (e.g., shape–color token pairs) must be detected robustly. Traditional deep learning models often overfit to superficial patterns, whereas symbolic approaches can offer improved interpretability. Our hybrid model bridges these paradigms with the following key components:

- **Transformer Encoder:** Processes input sequences of tokens (e.g., `(shape, color)` pairs) to generate contextual embeddings.  
- **Rule Extraction Module:** A shallow multi-layer perceptron (MLP) predicts binary predicate signals (e.g., `p_shape` and `p_color`) based on pooled contextual embeddings.
- **Differentiable Logic Layer:** Fuses the predicate signals with a linear transformation; subsequently, a simple averaging mechanism combines the deep and symbolic scores to deliver the final logit.
- **Dual-Loss Training:** Combines a primary classification loss (binary cross-entropy) on the final logit with an auxiliary loss that enforces fidelity on the predicted predicate signals (weighted at λ = 0.5).

## Model Architecture

The final decision is computed by:
  final_logit = (f_base(x) + f_logic(p_shape, p_color)) / 2

Where:
- **f_base(x):** The transformer-based classifier output.
- **f_logic(p_shape, p_color):** The output of the differentiable logic layer that processes predicate signals.

## Repository Structure

```
├── README.md
├── src
│   ├── model.py         # Contains implementations for the Transformer, MLP rule extractor, and logic fusion layer.
│   ├── training.py      # Training scripts with dual-loss formulation.
│   ├── evaluation.py    # Evaluation scripts to compute metrics including Shape-Weighted Accuracy (SWA).
│   └── utils.py         # Utility functions, including data preprocessing and metric calculations.
├── data
│   └── synthetic_dataset/  # Synthetic SPR dataset with token sequences (shape–color pairs).
└── experiments
    ├── config.yaml      # Configuration file with model hyperparameters and training settings.
    └── results/         # Directory to store training logs, model checkpoints, and evaluation plots.
```

## Experimental Setup

- **Dataset:**  
  A synthetic dataset containing shape–color token sequences.  
  - Train: 1,000 examples  
  - Development: 300 examples  
  - Test: 300 examples  
  Sequences are padded to a uniform length of 6 tokens.

- **Key Hyperparameters:**
  - Embedding Dimension: 32  
  - Attention Heads: 4  
  - Transformer Layers: 2  
  - Auxiliary Loss Weight (λ): 0.5  
  - Learning Rate: 1e-3  

- **Evaluation Metric:**  
  Shape-Weighted Accuracy (SWA) is used to evaluate the performance. The SWA metric weights correctly classified examples by their shape complexity.

## Getting Started

### Prerequisites

- Python 3.7 or later
- Recommended packages: PyTorch, NumPy, tqdm, pyyaml, and matplotlib (for plotting).

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/neuro-symbolic-spr.git
cd neuro-symbolic-spr
```

### Running the Experiments

1. **Data Preparation:**  
   Ensure the synthetic dataset is available in the `data/synthetic_dataset/` directory. Modify the data paths in `config.yaml` if necessary.

2. **Training the Model:**  
   You can train the model using the provided training script. For example:

   ```bash
   python src/training.py --config experiments/config.yaml
   ```

3. **Evaluating the Model:**  
   After training, run the evaluation script to compute the SWA and other metrics:

   ```bash
   python src/evaluation.py --config experiments/config.yaml
   ```

4. **Logging and Checkpoints:**  
   Checkpoints and training logs are stored in the `experiments/results/` directory. Figures for training loss and development accuracy can be found as "Figure_1.png" and "Figure_2.png" respectively.

## Results

Our preliminary experiments demonstrate that:

- **Baseline Transformer-only Model:**  
  - Test SWA: 0.6244

- **Hybrid Neuro-Symbolic Model:**  
  - Test SWA: 0.6444

- **Assumed State-of-the-Art (SOTA):**  
  - SWA: 0.8000

The improvement, although modest (~2% absolute increase in SWA), suggests that integrating explicit symbolic cues with deep contextual representations may help mitigate over-reliance on superficial features.

## Future Work

Future directions for this project include:
- Extending training to longer epochs and larger datasets for further improvements.
- Exploring more expressive differentiable logic formulations (e.g., multi-layer perceptron based logic modules, fuzzy logic techniques).
- Conducting systematic hyperparameter sweeps and ablation studies to isolate individual component contributions.
- Evaluating complementary metrics (e.g., Color-Weighted Accuracy) and fairness assessments across varied data distributions.

## Citation

If you find this work useful in your research, please consider citing our paper:

```
@article{neurosymbolic_spr,
  title={Neuro-Symbolic Integration for Symbolic Pattern Recognition},
  author={Agent Laboratory},
  journal={Preprint},
  year={2023},
  note={Available on arXiv or similar repositories}
}
```

## Contributing

Contributions and improvements are welcome! Feel free to open issues and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Happy coding and research!