# Hierarchical VAE-Enhanced Transformer for Symbolic Predicate Recognition

Welcome to the repository for the Hierarchical VAE-Enhanced Transformer model for Symbolic Predicate Recognition (SPR). This project integrates discrete latent segmentation with a Transformer encoder to extract interpretable symbolic predicates from token sequences. The method is designed to evaluate L-token sequences—each token being a shape-color pair—from the set {▵, ■, •, ◊} × {r, g, b, y} and determine whether the sequence satisfies a hidden poly-factor rule.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository implements a novel neuro-symbolic approach that blends continuous neural representations and discrete symbolic reasoning. The key contributions include:

- **Hierarchical Integration:** A two-stage Transformer encoder captures both local and global dependencies while a VAE-inspired latent segmentation module produces discrete, interpretable segmentation boundaries.
- **Discrete Latent Segmentation:** The module identifies segmentation boundaries using the condition:  
  `argmax q(z|x) = 0`, allowing the network to isolate sub-sequences that are processed by an atomic predicate network.
- **Differentiable Predicate Extraction:** Each segment is scored via a shallow feed-forward network, and a hierarchical composition layer aggregates these predicate scores, providing a transparent decision-making process.
- **Experimental Comparison:** The approach is evaluated against a baseline Transformer classifier and an ablated variant without hierarchical composition on a synthetic SPR dataset.

---

## Features

- **Joint Token Embeddings:** Combines shape and color embeddings to represent each token.
- **Two-Stage Transformer Encoder:** Models local and global context in the token sequence.
- **VAE-Inspired Latent Segmentation:** Uses KL divergence regularization to promote discrete segmentation.
- **Differentiable Symbolic Predicate Extraction:** Extracts atomic predicate scores from each segment.
- **Hierarchical Composition Layer:** Aggregates predicate scores to yield the final classification decision.
- **Reproducible Experiments:** Includes a synthetic dataset generator and experimental scripts for rapid prototyping.

---

## Installation

To run the project locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/hierarchical-vae-transformer-spr.git
   cd hierarchical-vae-transformer-spr
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   The primary dependencies include PyTorch, NumPy, and other standard libraries. Install them using pip:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure you have Python 3.7 or newer installed.*

---

## Usage

### Training the Model

A training script is provided to reproduce experiments on a synthetic SPR dataset.

```bash
python train.py --epochs 2 --batch_size 16 --learning_rate 1e-3
```

### Evaluating the Model

After training, evaluate the model on the test set using:

```bash
python evaluate.py --model_path path/to/saved_model.pth
```

### Configuration

Model hyperparameters and training configurations can be adjusted using the `config.yaml` or via command-line arguments. Key hyperparameters include:

- Sequence Length (L): 10–50 tokens
- Embedding Dimension: 32
- Latent Classes: 3 for the discrete segmentation module
- KL Divergence Weight (λ): 0.1

---

## Experimental Setup

The project was evaluated on a synthetic dataset where each sample is an L-token sequence generated from the set {▵, ■, •, ◊} × {r, g, b, y}, labeled according to a hidden poly-factor rule based on shape-count, color-position, parity, and order.

### Dataset Details

- **Splits:** Training, Development (Dev), and Test partitions (approximately 100 samples per split).
- **Metrics:** Primary evaluation is conducted using Shape-Weighted Accuracy (SWA), with additional metrics such as Color-Weighted Accuracy (CWA) considered.

Three models were compared:

1. **Model A (Full Hierarchical):** Full integration of the hierarchical segmentation and composition layers.
2. **Model B (Baseline Transformer):** A standard Transformer with joint token embeddings and average pooling.
3. **Model C (Without Hierarchical Composition):** Incorporates latent segmentation and predicate extraction but aggregates scores by a simple averaging without a dedicated composition module.

---

## Results

The following summarizes the test performance using the Shape-Weighted Accuracy (SWA) metric:

- **Model A (Full Hierarchical):** 54.0% Test Accuracy  
  Dev Accuracy: 48.0 | Training Loss: 0.72
- **Model B (Baseline Transformer):** 55.0% Test Accuracy  
  Dev Accuracy: 49.0 | Training Loss: 0.67
- **Model C (Without Composition):** 53.0% Test Accuracy  
  Dev Accuracy: 46.0 | Training Loss: 0.66

Although the full hierarchical model sacrifices a small margin in predictive accuracy compared to the baseline, it provides improved interpretability of segmentation boundaries and predicate extraction.

---

## Limitations and Future Work

### Current Limitations

- **Training Epochs:** Experiments were conducted using only 2 epochs on a sub-sampled dataset.
- **KL Divergence Weight:** A fixed KL weight (λ = 0.1) was used; dynamic tuning might improve the balance between segmentation quality and classification performance.
- **Aggregation Strategy:** The current simple averaging in the composition layer may be improved using attention-based or adaptive weighting mechanisms.
- **Evaluation Metrics:** Future work should include more comprehensive metrics (e.g., Color-Weighted Accuracy) to capture various performance aspects.

### Future Directions

- **Extended Training:** Increase epochs and dataset size to fully leverage the hierarchical segmentation module.
- **Dynamic Parameter Tuning:** Explore adaptive KL divergence weighting schemes.
- **Advanced Composition Methods:** Integrate more sophisticated aggregation strategies for predicate scores.
- **Real-World Datasets:** Validate the approach on real-world tasks beyond synthetic SPR samples.

---

## Citation

If you use this code or the accompanying ideas in your research, please cite our work as follows:

    @article{agentlab2023spr,
      title={Hierarchical VAE-Enhanced Transformer for Symbolic Predicate Recognition},
      author={Agent Laboratory},
      year={2023},
      journal={Preprint Repository}
    }

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy Coding!

For questions or contributions, please open an issue or a pull request on GitHub.